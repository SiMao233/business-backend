"""RAG 评测 CLI：跑检索层 / 答案层评测、落 JSON 报告、与基线对比、生成人工复核表（**只读**）。

用法（**必须在项目根目录执行**：`.env` 是相对路径；脚本内部也会 chdir 兜底）：

```bash
# 检索层评测（免 token，约 20 秒）
.venv/bin/python scripts/rag_eval.py
.venv/bin/python scripts/rag_eval.py --tag "chunk=1000/200"        # 给本次运行打标签

# 与基线对比：回答「这次改动 Recall@5 是升了还是降了」
.venv/bin/python scripts/rag_eval.py --baseline latest
.venv/bin/python scripts/rag_eval.py --baseline latest --fail-on-regression   # 有回退则退出码 1
.venv/bin/python scripts/rag_eval.py --save-baseline default                  # 固化基线

# 答案层评测（会真实调用模型）+ 人工复核表
.venv/bin/python scripts/rag_eval.py --answers --limit 10
.venv/bin/python scripts/rag_eval.py --answers --review-md evals/runs/review.md
.venv/bin/python scripts/rag_eval.py --review-load evals/runs/review.md
```

产物：
- 每次运行的完整 JSON → `evals/runs/eval-<时间戳>.json`（已 gitignore）；
- 基线 → `evals/baselines/<name>.json`（**要提交**，这样跨会话/跨机器可比）。

退出码：`0` 正常；`1` 阈值未达标或（配合 `--fail-on-regression`）存在指标回退；`2` 环境/参数错误。
⚠️ 全程只读：不写数据库、不改索引；答案层也不落会话/消息/用量记录。
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # .env 为相对路径，统一在项目根目录运行

from loguru import logger  # noqa: E402

from app.core.database import AsyncSessionLocal, engine  # noqa: E402
from app.modules.ai.eval_runner import run_answers_eval, run_retrieval_eval  # noqa: E402
from app.modules.ai.evaluation import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    EvalReport,
    load_golden_file,
    parse_review_markdown,
    render_diff,
    render_review_markdown,
    render_summary,
    summarize_review,
    threshold_to_metric,
)

RUNS_DIR = PROJECT_ROOT / "evals" / "runs"
BASELINES_DIR = PROJECT_ROOT / "evals" / "baselines"


def _fail(message: str, code: int = 2) -> None:
    print(f"❌ {message}", file=sys.stderr)
    raise SystemExit(code)


def _parse_ks(raw: str) -> list[int]:
    """把 `--k 1,3,5,10` 解析成整数列表（用 `type=` 交给 argparse，保证默认值也是整数）。"""
    try:
        values = sorted({int(item) for item in raw.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"需要逗号分隔的整数，收到 {raw!r}") from exc
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("至少需要一个正整数")
    return values


def _resolve_baseline(value: str) -> Path:
    """`latest` → runs 目录里最新一份；其它值 → baselines/<name>.json 或直接当路径。"""
    if value == "latest":
        candidates = sorted(RUNS_DIR.glob("eval-*.json"))
        if not candidates:
            _fail(f"没有可用的历史运行（{RUNS_DIR} 为空）")
        return candidates[-1]
    by_name = BASELINES_DIR / f"{value}.json"
    if by_name.is_file():
        return by_name
    direct = Path(value)
    if direct.is_file():
        return direct
    _fail(f"基线不存在：{value}（既不是 {by_name} 也不是文件路径）")
    raise  # pragma: no cover


def _threshold_failures(report: EvalReport, thresholds: dict[str, float]) -> list[str]:
    """按数据集的 `[thresholds]` 校验本次运行（与 pytest 门禁同一口径）。"""
    overall = (report.retrieval or {}).get("overall") or {}
    failures: list[str] = []
    for key, value in thresholds.items():
        metric = threshold_to_metric(key)
        measured = overall.get(metric)
        if measured is None:
            failures.append(f"{key} → {metric}: 报告里没有该指标（键写错了？）")
        elif measured < value:
            failures.append(f"{key} → {metric}: 实测 {measured:.3f} < 阈值 {value:.3f}")
    return failures


async def _run(args: argparse.Namespace, golden) -> EvalReport:
    runner = run_answers_eval if args.answers else run_retrieval_eval
    async with AsyncSessionLocal() as db:
        return await runner(
            db,
            golden,
            ks=tuple(args.k),
            tag=args.tag,
            limit=args.limit,
            categories=args.category or None,
            top_k=args.top_k,
            dataset_path=args.dataset,
            kb_override=args.kb,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评测（检索层 / 答案层，只读）")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Golden Dataset 路径")
    parser.add_argument("--kb", default=None, help="覆盖数据集里的知识库 ID（用于新语料试跑）")
    parser.add_argument(
        "--k", type=_parse_ks, default=[1, 3, 5, 10], help="Recall@K 的 K 列表（默认 1,3,5,10）"
    )
    parser.add_argument("--top-k", type=int, default=10, help="评测时临时使用的 top_k（默认 10）")
    parser.add_argument("--answers", action="store_true", help="跑答案层（真实调用模型，花 token）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（快速子集）")
    parser.add_argument("--category", action="append", default=[], help="只跑指定分类（可重复）")
    parser.add_argument("--tag", default="", help="本次运行的标签（如 chunk=1000/200）")
    parser.add_argument("--out", default=None, help="报告输出路径（默认 evals/runs/eval-<ts>.json）")
    parser.add_argument("--baseline", default=None, help="对比基线：latest | <名称> | <路径>")
    parser.add_argument("--save-baseline", default=None, help="把本次结果固化为基线（名称）")
    parser.add_argument("--review-md", default=None, help="生成人工复核表（Markdown）到指定路径")
    parser.add_argument("--review-load", default=None, help="回读已填写的人工复核表并汇总")
    parser.add_argument(
        "--fail-on-regression", action="store_true", help="存在指标回退时以退出码 1 结束"
    )
    parser.add_argument("--quiet", action="store_true", help="静默库日志（评测管线默认已压到 WARNING）")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR" if args.quiet else "WARNING")

    if args.review_load:
        return _review_load(args.review_load)

    golden = load_golden_file(args.dataset)
    try:
        report = asyncio.run(_run(args, golden))
    except Exception as exc:  # noqa: BLE001 - 环境问题要给可读提示，而不是裸 traceback
        _fail(
            f"评测执行失败：{exc!r}\n"
            "  常见原因：远程 Qdrant / MySQL 不可达、embedding 服务超时、模型网关不可用。\n"
            "  排查：.venv/bin/python scripts/rag_index_audit.py  （索引对账，只读）"
        )

    print(render_summary(report, ks=tuple(args.k)))

    out_path = _write_report(args, report)
    print(f"\n报告已写入：{out_path}")

    if args.review_md:
        review_path = Path(args.review_md)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(render_review_markdown(report), encoding="utf-8")
        print(f"人工复核表已写入：{review_path}（填完用 --review-load 汇总）")

    if args.save_baseline:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = BASELINES_DIR / f"{args.save_baseline}.json"
        baseline_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"基线已固化：{baseline_path}")

    exit_code = 0
    failures = _threshold_failures(report, golden.thresholds)
    if failures:
        print("\n❌ 阈值未达标：")
        for item in failures:
            print(f"  - {item}")
        exit_code = 1
    else:
        print("\n✅ 阈值全部达标" if golden.thresholds else "\n⚠️ 数据集未配置 [thresholds]，跳过门禁")

    if args.baseline:
        diff_path = _resolve_baseline(args.baseline)
        baseline = EvalReport.from_dict(json.loads(diff_path.read_text(encoding="utf-8")))
        diff = compare_and_render(baseline, report, question_metric=f"hit@{args.k[-1]}")
        print(f"\n基线：{diff_path}")
        print(diff)
        if diff and args.fail_on_regression and _has_regression(baseline, report):
            exit_code = 1

    asyncio.run(engine.dispose(close=False))
    return exit_code


def _write_report(args: argparse.Namespace, report: EvalReport) -> Path:
    if args.out:
        path = Path(args.out)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = RUNS_DIR / f"eval-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compare_and_render(baseline: EvalReport, current: EvalReport, *, question_metric: str) -> str:
    from app.modules.ai.evaluation import compare_reports

    return render_diff(compare_reports(baseline, current, question_metric=question_metric))


def _has_regression(baseline: EvalReport, current: EvalReport) -> bool:
    from app.modules.ai.evaluation import compare_reports

    return compare_reports(baseline, current).has_regression


def _review_load(path: str) -> int:
    file_path = Path(path)
    if not file_path.is_file():
        _fail(f"复核表不存在：{file_path}")
    verdicts = parse_review_markdown(file_path.read_text(encoding="utf-8"))
    summary = summarize_review(verdicts)
    print(f"人工复核汇总（{file_path}）")
    print(f"  已填写：{summary['filled']} 题")
    print(f"  通过 {summary['pass']} / 不通过 {summary['fail']} / 存疑 {summary['unsure']}")
    rate = summary["pass_rate"]
    print(f"  通过率：{'—' if rate is None else f'{rate:.1%}'}（未填题不计入分母）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
