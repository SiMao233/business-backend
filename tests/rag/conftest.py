"""`tests/rag/` 的公共夹具：环境探测 + Golden Dataset + 一次评测报告（会话内缓存）。

检索评测需要**真实环境**（MySQL + Qdrant + embedding 服务）。环境不可达时**不做静默通过**，
而是 `skip` 并打印可执行的排查提示 —— 否则在离线机器上会被误读成「评测通过」。

答案层评测额外需要**模型网关**（花 token），因此默认关闭：
只有设了 `RAG_EVAL_ANSWERS=1` 才会跑，题量由 `RAG_EVAL_ANSWER_LIMIT` 控制（默认 10，0 = 全部）。

报告在一次 pytest 会话里只跑一遍（40 题各一次 embedding 调用，约十几秒），多个用例共享同一份，
避免重复等待与重复计费；`-s` 下会把指标摘要直接打印出来。
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal, engine
from app.modules.ai.eval_runner import run_answers_eval, run_retrieval_eval
from app.modules.ai.evaluation import DEFAULT_DATASET_PATH, EvalReport, GoldenFile, load_golden_file, render_summary

# 会话级缓存（夹具是 function scope，因为 pytest-asyncio 的 loop scope 是 function）
_PROBE: dict[str, str | None] = {}
_REPORTS: dict[str, EvalReport] = {}

# 报告缓存键要包含影响检索的开关，否则会话内改了配置会拿到过期报告
_CACHE_KEYS = (
    "rag_hybrid_enabled",
    "rag_rerank_enabled",
    "rag_query_rewrite_enabled",
    "rag_top_k",
    "chunk_size",
    "chunk_overlap",
)


@pytest.fixture(scope="session")
def golden() -> GoldenFile:
    """Golden Dataset（含阈值）。数据集本身的问题由 test_golden_dataset.py 负责报错。"""
    return load_golden_file(DEFAULT_DATASET_PATH)


async def _env_problem(golden: GoldenFile) -> str | None:
    """返回环境不可用的原因；None 表示可用。"""
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
            ids = ", ".join(f"'{dataset.knowledge_base_id}'" for dataset in golden.datasets)
            count = (
                await db.execute(
                    text(f"SELECT COUNT(*) FROM sys_knowledge_chunk WHERE knowledge_base_id IN ({ids})")
                )
            ).scalar()
    except Exception as exc:  # noqa: BLE001 - 任何连接/权限/网络问题都归为「环境不可用」
        return f"RAG 评测需要真实 MySQL，当前不可用：{exc!r}"

    if not count:
        return "目标知识库在 MySQL 里没有任何切块（可能未索引/已被删除）"

    try:
        from app.modules.ai.vector import get_qdrant_client

        await get_qdrant_client().get_collections()
    except Exception as exc:  # noqa: BLE001 - 同上
        return f"RAG 评测需要 Qdrant，当前不可用：{exc!r}"
    return None


async def _dispose_pool() -> None:
    """归还连接池：**pytest 每个用例一个事件循环**，而池里的连接绑定在创建它的 loop 上。

    不 dispose 的话，下一个用例（新 loop）会拿到上个 loop 的连接，
    报 `RuntimeError: Event loop is closed` —— 实测踩过一次（新增夹具链后探测与使用落在不同 loop）。
    `close=False` 避免关闭时的报错噪音。
    """
    await engine.dispose(close=False)


@pytest.fixture(autouse=True)
def _reset_qdrant_singleton() -> Iterator[None]:
    """每个用例结束后丢弃 Qdrant 客户端单例（与 `_dispose_pool` 同一类问题的另一半）。

    `app/modules/ai/vector.py::get_qdrant_client()` 带 `@lru_cache`，是**进程级单例**，
    而 pytest-asyncio 的 loop scope 是 function（每个用例一个新事件循环）→ 后面的用例会复用
    绑定在已关闭 loop 上的 httpx 连接池，报 `RuntimeError: Event loop is closed` /
    `ResponseHandlingException`（`pytest -q` 与 `pytest tests/rag/ -q` 稳定复现 2 个 setup error，
    而单跑 `test_retrieval_eval.py` 却能过，极易误判为「偶发」）。
    每个用例结束即清缓存，下一个用例就会在自己的 loop 里新建客户端。
    """
    yield
    # 延迟导入：保证离线机器上（未装 qdrant 相关依赖链）仍能收集本文件
    from app.modules.ai.vector import get_qdrant_client

    get_qdrant_client.cache_clear()


@pytest.fixture
async def rag_env(golden: GoldenFile) -> None:
    """探测真实环境；不可用则 skip（附带排查命令）。"""
    if "problem" not in _PROBE:
        _PROBE["problem"] = await _env_problem(golden)
        await _dispose_pool()
    if _PROBE["problem"]:
        pytest.skip(
            f"{_PROBE['problem']}\n"
            "  排查：.venv/bin/python scripts/rag_index_audit.py  （索引对账，只读）"
        )
    return None


@pytest.fixture
async def eval_report(golden: GoldenFile, rag_env: Any) -> EvalReport:
    """跑一次检索评测并缓存；`-s` 时打印指标摘要。"""
    settings = get_settings()
    key = f"{DEFAULT_DATASET_PATH}|{golden.version}|" + ",".join(
        f"{name}={getattr(settings, name, None)}" for name in _CACHE_KEYS
    )
    if key not in _REPORTS:
        async with AsyncSessionLocal() as db:
            _REPORTS[key] = await run_retrieval_eval(
                db, golden, dataset_path=Path(DEFAULT_DATASET_PATH)
            )
        await _dispose_pool()
        print("\n" + render_summary(_REPORTS[key]) + "\n")
    return _REPORTS[key]


def _answers_enabled() -> bool:
    """答案层默认不跑：会真实调用模型（花 token），需要显式开启。"""
    return os.getenv("RAG_EVAL_ANSWERS", "").strip().lower() in {"1", "true", "yes", "on"}


def _answers_limit() -> int | None:
    """答案层题量：`RAG_EVAL_ANSWER_LIMIT`（默认 10；≤ 0 表示全部）。"""
    raw = os.getenv("RAG_EVAL_ANSWER_LIMIT", "10").strip() or "10"
    value = int(raw)
    return None if value <= 0 else value


@pytest.fixture
async def answers_report(golden: GoldenFile, rag_env: Any) -> EvalReport:
    """答案层评测报告（含检索层指标）；需 `RAG_EVAL_ANSWERS=1` 显式启用。"""
    if not _answers_enabled():
        pytest.skip(
            "答案层评测会真实调用模型（花 token），默认跳过。启用："
            "RAG_EVAL_ANSWERS=1 .venv/bin/pytest tests/rag/ -q"
            "（可用 RAG_EVAL_ANSWER_LIMIT 控制题量，默认 10，0 = 全部）"
        )
    limit = _answers_limit()
    settings = get_settings()
    key = f"answers|{limit}|{DEFAULT_DATASET_PATH}|" + ",".join(
        f"{name}={getattr(settings, name, None)}" for name in _CACHE_KEYS
    )
    if key not in _REPORTS:
        async with AsyncSessionLocal() as db:
            _REPORTS[key] = await run_answers_eval(
                db,
                golden,
                limit=limit,
                tag="pytest-answers",
                dataset_path=Path(DEFAULT_DATASET_PATH),
            )
        await _dispose_pool()
        print("\n" + render_summary(_REPORTS[key]) + "\n")
    return _REPORTS[key]
