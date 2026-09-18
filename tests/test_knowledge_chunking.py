"""结构感知切分测试：章节上下文、结构完整性、chunk_size 兼容、中文文档。

纯函数测试，不依赖 MySQL / Qdrant / ARQ。
"""

import pytest

from app.common.enums import ChunkType
from app.core.config import get_settings
from app.modules.knowledge.parser import (
    BlockKind,
    ChunkDraft,
    ParsedBlock,
    parse_document,
    split_blocks,
)


def _set_chunk_size(monkeypatch: pytest.MonkeyPatch, size: int, overlap: int = 0) -> None:
    """覆盖全局切分配置（get_settings 是 lru_cache，patch 的是同一个实例）。"""
    settings = get_settings()
    monkeypatch.setattr(settings, "chunk_size", size)
    monkeypatch.setattr(settings, "chunk_overlap", overlap)


# ---- section_path：章节上下文 ----


def test_section_path_tracks_heading_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_chunk_size(monkeypatch, 1000)
    md = "# 员工手册\n\n## 一、总则\n\n本手册适用于全体员工。\n\n## 二、考勤\n\n每日打卡两次。"
    drafts = split_blocks(parse_document(md.encode(), "md"))
    assert [d.section_path for d in drafts] == [
        "员工手册 > 一、总则",
        "员工手册 > 二、考勤",
    ]
    # 标题与其内容同块（标题不孤立成块）
    assert drafts[0].content.startswith("一、总则")
    assert "本手册适用于全体员工。" in drafts[0].content


def test_section_path_repeats_on_every_chunk_of_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一章节被切成多块时，每块都带同一章节路径（切片后不丢上下文）。"""
    _set_chunk_size(monkeypatch, 60)
    md = "# 员工手册\n\n## 二、考勤\n\n" + "打卡规则说明。" * 20
    drafts = split_blocks(parse_document(md.encode(), "md"))
    assert len(drafts) > 2
    assert {d.section_path for d in drafts} == {"员工手册 > 二、考勤"}


def test_heading_is_not_emitted_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """连续标题不产出「只有标题」的碎块（标题跟随其内容）。"""
    _set_chunk_size(monkeypatch, 1000)
    md = "# 员工手册\n\n## 二、考勤\n\n每日打卡两次。"
    drafts = split_blocks(parse_document(md.encode(), "md"))
    assert len(drafts) == 1
    assert drafts[0].section_path == "员工手册 > 二、考勤"
    assert drafts[0].content.startswith("二、考勤")
    assert drafts[0].content.endswith("每日打卡两次。")


def test_section_path_is_none_without_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_chunk_size(monkeypatch, 1000)
    drafts = split_blocks(parse_document("第一段中文。\n\n第二段中文。".encode(), "txt"))
    assert len(drafts) == 1
    assert drafts[0].section_path is None
    assert drafts[0].chunk_type == ChunkType.TEXT.value
    # 段落边界（空行）被保留
    assert drafts[0].content == "第一段中文。\n\n第二段中文。"


# ---- 结构完整性：列表 / 表格 ----


def test_list_not_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """列表整组保留（即使 chunk_size 只够勉强放下它）。"""
    _set_chunk_size(monkeypatch, 200)
    md = "## 清单\n\n- 第一项内容\n- 第二项内容\n- 第三项内容\n\n后续段落。"
    drafts = split_blocks(parse_document(md.encode(), "md"))
    list_chunk = next(d for d in drafts if "第一项内容" in d.content)
    assert "- 第一项内容" in list_chunk.content
    assert "- 第二项内容" in list_chunk.content
    assert "- 第三项内容" in list_chunk.content


def test_table_chunk_type_and_rows_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_chunk_size(monkeypatch, 1000)
    md = "| 姓名 | 部门 |\n| --- | --- |\n| 张三 | 研发 |\n| 李四 | 市场 |\n"
    drafts = split_blocks(parse_document(md.encode(), "md"))
    assert len(drafts) == 1
    assert drafts[0].chunk_type == ChunkType.TABLE.value
    assert drafts[0].content.splitlines() == md.strip().splitlines()


def test_oversized_table_splits_on_row_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """超大表格按整行切分，不把某一行数据劈开，且每个子表都重复表头。"""
    _set_chunk_size(monkeypatch, 60, overlap=0)
    header = ["| 姓名 | 部门 |", "| --- | --- |"]
    rows = [f"| 员工{i:02d} | 部门{i:02d} |" for i in range(20)]
    drafts = split_blocks([ParsedBlock(text="\n".join(header + rows), kind=BlockKind.TABLE)])
    assert len(drafts) > 1
    for draft in drafts:
        assert draft.chunk_type == ChunkType.TABLE.value
        lines = draft.content.splitlines()
        # 每个子表都以表头 + 分隔行开头（子表自解释，列含义不丢）
        assert lines[0] == header[0]
        assert lines[1] == header[1]
        # 数据行不被劈开
        for line in lines:
            assert line.startswith("|")
            assert line.endswith("|")
    # 所有数据行都出现且仅出现一次
    joined = "\n".join(d.content for d in drafts)
    for row in rows:
        assert joined.count(row) == 1


def test_oversized_table_without_separator_keeps_first_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DOCX 等无分隔行的表格：重复首行作为表头。"""
    _set_chunk_size(monkeypatch, 40, overlap=0)
    lines = ["项目 | 标准"] + [f"项目{i:02d} | 标准{i:02d}" for i in range(10)]
    drafts = split_blocks([ParsedBlock(text="\n".join(lines), kind=BlockKind.TABLE)])
    assert len(drafts) > 1
    assert all(d.content.splitlines()[0] == "项目 | 标准" for d in drafts)


def test_oversized_table_after_heading_takes_table_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """标题 + 超长表格（实际文档里最常见的形态）也必须走表头重复路径。

    回归：此前只有「纯表格」才会重复表头，带标题时类型为 mixed 会走通用字符切分，
    导致第 2 片起只有数据行、丢失表头与列含义。
    """
    _set_chunk_size(monkeypatch, 90, overlap=0)
    header = ["| 项目 | 标准 |", "| --- | --- |"]
    rows = [f"| 项目{i:02d} | 标准{i:02d} |" for i in range(10)]
    blocks = [
        ParsedBlock(text="2.1 打卡规则", kind=BlockKind.HEADING, level=3),
        ParsedBlock(text="\n".join(header + rows), kind=BlockKind.TABLE),
    ]
    drafts = split_blocks(blocks)
    assert len(drafts) > 1
    assert all(d.section_path == "2.1 打卡规则" for d in drafts)
    # 首片：标题 + 表头（保留 mixed）
    assert drafts[0].content.startswith("2.1 打卡规则\n\n")
    assert header[0] in drafts[0].content
    assert drafts[0].chunk_type == ChunkType.MIXED.value
    # 其余片：纯表格，且都重复表头
    for draft in drafts[1:]:
        assert draft.chunk_type == ChunkType.TABLE.value
        assert draft.content.splitlines()[0] == header[0]
        assert draft.content.splitlines()[1] == header[1]
    # 数据行不丢不重
    joined = "\n".join(d.content for d in drafts)
    for row in rows:
        assert joined.count(row) == 1


def test_oversized_block_fallback_keeps_section_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """单个结构块超长时回退字符切分，但章节路径与标题文本不丢。"""
    _set_chunk_size(monkeypatch, 50)
    md = "# 员工手册\n\n" + "很长的正文内容。" * 30
    drafts = split_blocks(parse_document(md.encode(), "md"))
    assert len(drafts) > 1
    assert all(d.section_path == "员工手册" for d in drafts)
    assert "员工手册" in drafts[0].content
    # 拼接后正文内容不丢失（回退切分点不同，不做逐字比对）
    joined = "".join(d.content for d in drafts)
    assert "很长的正文内容。" * 5 in joined


# ---- chunk_size / chunk_overlap 仍然生效 ----


def test_chunk_size_respected_for_long_paragraph(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_chunk_size(monkeypatch, 80, overlap=0)
    drafts = split_blocks([ParsedBlock(text="这是一句话。" * 60)])
    assert len(drafts) > 1
    assert all(len(d.content) <= 80 for d in drafts)


def test_chunk_overlap_still_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    """overlap 配置生效（回退字符切分路径）：带重叠时块数不少于无重叠。"""
    base = "".join(f"第{i}句测试文本内容。" for i in range(40))
    _set_chunk_size(monkeypatch, 100, overlap=0)
    no_overlap = split_blocks([ParsedBlock(text=base)])
    _set_chunk_size(monkeypatch, 100, overlap=30)
    with_overlap = split_blocks([ParsedBlock(text=base)])
    assert len(with_overlap) >= len(no_overlap)


# ---- 分页 ----


def test_pdf_page_change_forces_new_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF 换页必须断开，保证 page_no 精确。"""
    _set_chunk_size(monkeypatch, 1000)
    blocks = [
        ParsedBlock(text="第一页内容。", page_no=1),
        ParsedBlock(text="第二页内容。", page_no=2),
    ]
    drafts = split_blocks(blocks)
    assert [d.page_no for d in drafts] == [1, 2]


# ---- 嵌入文本 ----


def test_embedding_text_prepends_section_path() -> None:
    draft = ChunkDraft(content="每日打卡两次。", section_path="员工手册 > 二、考勤")
    assert draft.embedding_text == "员工手册 > 二、考勤\n每日打卡两次。"
    # 无章节时与正文一致，避免无谓前缀
    assert ChunkDraft(content="正文").embedding_text == "正文"


def test_empty_blocks_yield_no_drafts() -> None:
    assert split_blocks([]) == []
