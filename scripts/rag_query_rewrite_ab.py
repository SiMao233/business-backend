"""Query Rewrite 真实 A/B 与降级演练（需要真实 MySQL + Qdrant + 模型网关）。

用途：调整 RAG 参数（改写开关 / 超时、`rag_top_k`、Reranker 开关…）后，验证
「多轮指代问题」的改写是否**真的**改善了检索，以及各降级路径是否仍然正常。

四部分：
- PART 1/2  多轮场景：真实改写 → 原 query 与改写 query 的检索对比（含**重复条目检查**，
            用于验证融合键口径正确 —— 两路命中同一 chunk 必须合并成一条）；
- PART 3    短路对照：无上下文 / 自足问题 / 开关关闭 → 必须**零 LLM 调用**；
- PART 4    降级演练：开关关闭 / 极短超时（验证未触发 SDK 重试）/ 网关不可达。

⚠️ 会真实调用模型网关（产生少量 token 费用）与远程 Qdrant；**全程只读**，不写任何数据。

内置场景针对两类主题（「员工手册 / 考勤」「HTTP / HTTPS」）—— 它们需要 `--kb` 指向的
知识库里**确实有**对应文档才有验证意义；不匹配的场景会被自动跳过（不会误报失败）。

用法（**必须在项目根目录执行**：`.env` 是相对路径）：
    .venv/bin/python scripts/rag_query_rewrite_ab.py --kb <kb_id>
    .venv/bin/python scripts/rag_query_rewrite_ab.py --kb <kb_id> --scenario 1

退出码：0 = 全部通过；1 = 有断言失败（或没有任何场景可跑）。
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # .env 为相对路径，统一在项目根目录运行

from sqlalchemy import text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.database import AsyncSessionLocal  # noqa: E402
from app.modules.ai.query import RewriteSource, apply_query_rewrite, build_rewrite_llm  # noqa: E402
from app.modules.ai.retrieval import retrieve_hits  # noqa: E402

# 内置场景：历史（user/assistant 成对）+ 指代式追问 + 期望命中的文档名片段
SCENARIOS = [
    {
        "name": "员工手册·考勤（指代）",
        "history": [
            ("user", "员工手册里的考勤制度是怎么规定的？"),
            (
                "assistant",
                "工作时间为周一至周五上午九点到下午六点，午休一小时；"
                "迟到或早退超过三十分钟按半天事假处理。",
            ),
        ],
        "query": "那这个怎么办？",
        "expect_doc": "员工手册",
    },
    {
        "name": "HTTP/HTTPS（指代）",
        "history": [
            ("user", "HTTP 和 HTTPS 有什么区别？"),
            ("assistant", "HTTPS 在 HTTP 之上加了 TLS 加密，默认端口 443，能防窃听与篡改。"),
        ],
        "query": "它主要解决什么问题？",
        "expect_doc": "HTTP",
    },
]

failures: list[str] = []


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    if not ok:
        failures.append(f"{label}: got={got!r} want={want!r}")
    print(f"    {'✅' if ok else '❌'} {label}: {got!r}")


class CountingLlm:
    """计数包装：用于断言「短路时零 LLM 调用」。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        return await self.inner.ainvoke(messages, **kwargs)


async def pick_chat_model() -> tuple[str, str | None, str | None]:
    """取一个启用的 chat 模型（与 `chat/service._build_llm` 同源：优先 opencode 网关）。"""
    async with AsyncSessionLocal() as db:
        row = (
            await db.execute(
                text(
                    "SELECT i.code, p.api_key, p.base_url "
                    "FROM sys_model_instance i JOIN sys_model_provider p ON p.id = i.provider_id "
                    "WHERE i.status = 1 AND p.status = 1 AND i.model_type = 'chat' "
                    "AND p.base_url LIKE '%opencode%' ORDER BY i.create_time LIMIT 1"
                )
            )
        ).first()
    if row is None:
        raise SystemExit("库里没有启用的 chat 模型实例，无法验证")
    return row[0], row[1], row[2]


async def kb_document_names(kb_id: str) -> list[str]:
    """该知识库的文档名列表。

    ⚠️ `kb_id` 必须是 **hex（无连字符）**：MySQL 的 Uuid 列与 Qdrant payload 都存 hex，
    传带连字符的 UUID 字符串会一条也匹配不到。
    """
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                text("SELECT name FROM sys_knowledge_document WHERE knowledge_base_id = :kb"),
                {"kb": kb_id},
            )
        ).all()
    return [row[0] or "" for row in rows]


def show_hits(hits: list[dict], limit: int = 4) -> None:
    if not hits:
        print("      （0 条命中）")
        return
    for i, hit in enumerate(hits[:limit], start=1):
        rerank = hit.get("rerank_score")
        score = "None" if hit.get("score") is None else f"{hit['score']:.4f}"
        rerank_text = "" if rerank is None else f" rerank={rerank:.4f}"
        content = (hit.get("content") or "").replace("\n", " ")[:32]
        print(
            f"      {i}. doc={hit.get('document_name')!r} score={score}{rerank_text} "
            f"from={hit.get('retrievers')} chunk={str(hit.get('chunk_id'))[:12]}  {content!r}"
        )


def dup_check(label: str, hits: list[dict]) -> None:
    """融合键口径正确时，同一内容不应出现两条（旧方案切块曾因此重复）。"""
    contents = [hit.get("content") for hit in hits]
    dupes = len(contents) - len(set(contents))
    print(f"      重复条目数={dupes}（总数={len(contents)}，去重后={len(set(contents))}）")
    check(f"{label} 无重复条目", dupes, 0)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Query Rewrite 真实 A/B 与降级演练")
    parser.add_argument("--kb", required=True, help="知识库 ID（hex 或带连字符）")
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help=f"只跑指定场景（0~{len(SCENARIOS) - 1}）；不传则跑全部",
    )
    args = parser.parse_args()

    # ⚠️ 统一用 hex（无连字符）：MySQL 的 Uuid 列与 Qdrant payload 都是 hex 形态，
    # 带连字符的写法在直接 SQL 比对时会一条都匹配不到（retrieve_hits 内部会 UUID() 归一化，两者都收）
    kb_id = UUID(args.kb).hex
    config = {"knowledge_ids": [kb_id]}
    settings = get_settings()
    model_code, api_key, base_url = await pick_chat_model()
    headers = {"x-opencode-session": "qrab01"} if base_url and "opencode" in base_url else None

    def real_llm():
        return build_rewrite_llm(
            model_code, api_key=api_key, base_url=base_url, default_headers=headers
        )

    print(
        f"[环境] model={model_code} kb={kb_id}\n"
        f"       改写开关={settings.rag_query_rewrite_enabled} "
        f"超时={settings.rag_query_rewrite_timeout}s "
        f"rag_top_k={settings.rag_top_k} Reranker={settings.rag_rerank_enabled}"
    )

    doc_names = await kb_document_names(kb_id)
    scenarios = SCENARIOS if args.scenario is None else [SCENARIOS[args.scenario]]
    applicable = [
        sc for sc in scenarios if any(sc["expect_doc"] in name for name in doc_names)
    ]
    for sc in scenarios:
        if sc not in applicable:
            print(
                f"\n⚠️ 跳过场景「{sc['name']}」：该知识库里没有名称含 {sc['expect_doc']!r} 的文档\n"
                f"   （该 KB 的文档：{doc_names}）"
            )
    if not applicable:
        print("\n❌ 没有任何场景适用于该知识库，请用 --kb 指定合适的库")
        return 1

    # ---------------- PART 1 + 2：真实改写 + 检索 A/B ----------------
    settings.rag_query_rewrite_enabled = True
    settings.rag_rerank_enabled = False

    async with AsyncSessionLocal() as db:
        for sc in applicable:
            print(f"\n{'=' * 78}\nPART 1/2 场景：{sc['name']}\n{'=' * 78}")
            counter = CountingLlm(real_llm())
            started = time.perf_counter()
            result = await apply_query_rewrite(counter, sc["query"], sc["history"])
            wall = (time.perf_counter() - started) * 1000

            print(f"  原问题   : {result.original_query!r}")
            print(f"  改写后   : {result.rewritten_query!r}")
            print(
                f"  rewritten={result.rewritten}  source={result.source}  "
                f"elapsed={result.elapsed_ms}ms（实测墙钟 {wall:.0f}ms）"
            )
            if result.source == RewriteSource.FAILURE:
                print(
                    f"    ⚠️ 改写降级（多为超时，不是代码 bug）：当前 "
                    f"RAG_QUERY_REWRITE_TIMEOUT={settings.rag_query_rewrite_timeout}s，"
                    f"推理模型的改写耗时波动大（实测 1.3~6.4s）。\n"
                    f"       要么调大该超时（最坏 ttft 同步增加），要么接受降级"
                    f"（行为等价于未启用改写，安全但拿不到收益）。"
                )
            check("改写生效（source=MODEL）", result.source, RewriteSource.MODEL)
            check("rewritten=True", result.rewritten, True)
            check("改写结果不等于原问题", result.rewritten_query != sc["query"], True)
            check("LLM 只被调用 1 次", counter.calls, 1)

            print("\n  --- 检索 A/B（纯 Hybrid，Reranker 关闭）---")
            print("  ① 用【原 query】检索：")
            hits_orig = await retrieve_hits(db, config, result.original_query)
            show_hits(hits_orig)
            print("  ② 用【改写后 query】检索：")
            hits_new = await retrieve_hits(db, config, result.rewritten_query)
            show_hits(hits_new)

            print(f"\n  命中数：原={len(hits_orig)} → 改写后={len(hits_new)}")
            dup_check("纯 Hybrid（原 query）", hits_orig)
            dup_check("纯 Hybrid（改写后）", hits_new)

            top_orig = hits_orig[0].get("document_name") if hits_orig else ""
            top_new = hits_new[0].get("document_name") if hits_new else ""
            print(f"  首位文档：原={top_orig!r} → 改写后={top_new!r}")
            check("改写后首位命中目标文档", sc["expect_doc"] in (top_new or ""), True)
            check(
                "改写明显优于原 query",
                sc["expect_doc"] in (top_new or "")
                and sc["expect_doc"] not in (top_orig or ""),
                True,
            )

            print("\n  --- 检索 A/B（Hybrid + Reranker 全链路）---")
            settings.rag_rerank_enabled = True
            try:
                full_orig = await retrieve_hits(db, config, result.original_query)
                full_new = await retrieve_hits(db, config, result.rewritten_query)
                print("  ① 原 query：")
                show_hits(full_orig)
                print("  ② 改写后 query：")
                show_hits(full_new)
                dup_check("全链路（改写后）", full_new)
                top_full = full_new[0].get("document_name") if full_new else ""
                check("全链路改写后首位命中目标文档", sc["expect_doc"] in (top_full or ""), True)
            finally:
                settings.rag_rerank_enabled = False

    # ---------------- PART 3：短路对照 ----------------
    print(f"\n{'=' * 78}\nPART 3 短路对照（必须零 LLM 调用）\n{'=' * 78}")
    cases = [
        ("无上下文（首轮）", "那这个怎么办？", [], True, RewriteSource.NO_CONTEXT),
        (
            "简单明确问题（有历史）",
            "员工手册里的考勤制度是怎么规定的？",
            applicable[0]["history"],
            True,
            RewriteSource.NOT_NEEDED,
        ),
        ("开关关闭", "那这个怎么办？", applicable[0]["history"], False, RewriteSource.DISABLED),
    ]
    for label, query, history, enabled, expect_source in cases:
        settings.rag_query_rewrite_enabled = enabled
        counter = CountingLlm(real_llm())
        result = await apply_query_rewrite(counter, query, history)
        print(
            f"  {label}: source={result.source} rewritten={result.rewritten} "
            f"LLM 调用={counter.calls}"
        )
        check(f"  {label} source", result.source, expect_source)
        check(f"  {label} 零调用", counter.calls, 0)
        check(f"  {label} 返回原问题", result.rewritten_query, query)

    # ---------------- PART 4：降级演练 ----------------
    print(f"\n{'=' * 78}\nPART 4 降级演练\n{'=' * 78}")
    history = applicable[0]["history"]
    query = applicable[0]["query"]

    async with AsyncSessionLocal() as db:
        print("  (a) 开关关闭 → 行为与改造前一致")
        settings.rag_query_rewrite_enabled = False
        counter = CountingLlm(real_llm())
        result = await apply_query_rewrite(counter, query, history)
        check("    source=DISABLED", result.source, RewriteSource.DISABLED)
        check("    零 LLM 调用", counter.calls, 0)
        hits = await retrieve_hits(db, config, result.rewritten_query)
        check("    检索仍可用", isinstance(hits, list), True)
        print(f"    命中数={len(hits)}")

        print("  (b) 极短超时（0.001s）→ 必须快速降级（验证未触发 SDK 重试）")
        settings.rag_query_rewrite_enabled = True
        settings.rag_query_rewrite_timeout = 0.001
        counter = CountingLlm(real_llm())
        started = time.perf_counter()
        result = await apply_query_rewrite(counter, query, history)
        wall = (time.perf_counter() - started) * 1000
        print(f"    source={result.source} rewritten={result.rewritten} 墙钟={wall:.0f}ms")
        check("    source=FAILURE", result.source, RewriteSource.FAILURE)
        check("    回退原问题", result.rewritten_query, query)
        check("    尝试调用 1 次（未重试）", counter.calls, 1)
        check("    墙钟 < 1500ms（未触发 SDK 重试）", wall < 1500, True)
        hits = await retrieve_hits(db, config, result.rewritten_query)
        check("    检索仍可用", isinstance(hits, list), True)
        print(f"    命中数={len(hits)}")

        print("  (c) 网关不可达（base_url 指向黑洞端口）→ 回退原问题且不抛")
        unreachable = build_rewrite_llm(
            model_code, api_key="sk-x", base_url="http://127.0.0.1:9/v1"
        )
        counter = CountingLlm(unreachable)
        started = time.perf_counter()
        result = await apply_query_rewrite(counter, query, history)
        wall = (time.perf_counter() - started) * 1000
        print(f"    source={result.source} rewritten={result.rewritten} 墙钟={wall:.0f}ms")
        check("    source=FAILURE", result.source, RewriteSource.FAILURE)
        check("    回退原问题", result.rewritten_query, query)
        hits = await retrieve_hits(db, config, result.rewritten_query)
        check("    检索仍可用", isinstance(hits, list), True)
        print(f"    命中数={len(hits)}")

    print(f"\n{'=' * 78}")
    if failures:
        print(f"失败 {len(failures)} 项 ❌")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
