"""文档解析与切分测试（纯函数，不依赖 MySQL / Qdrant / ARQ）。"""

import io

import pytest
from docx import Document as DocxDocument

from app.core.config import get_settings
from app.core.exceptions import ValidateError
from app.modules.knowledge.parser import SUPPORTED_EXTS, parse_document, split_text


def _long_text(sentences: int = 200) -> str:
    """构造超过 chunk_size 的中文长文本（每句一个句号，便于按中文标点切分）。"""
    return "".join(f"这是第{i}句话，用于测试切分逻辑。" for i in range(sentences))


# ---- parse_document ----


def test_parse_txt() -> None:
    assert parse_document("你好，世界".encode(), "txt") == "你好，世界"


def test_parse_md() -> None:
    text = parse_document("# 标题\n\n正文内容".encode(), "md")
    assert text.startswith("# 标题")
    assert "正文内容" in text


def test_parse_unsupported_ext() -> None:
    # 文件层白名单比解析层宽（如 xlsx），解析层必须明确拒绝
    with pytest.raises(ValidateError):
        parse_document(b"data", "xlsx")
    with pytest.raises(ValidateError):
        parse_document(b"data", None)


def test_parse_docx_paragraphs() -> None:
    buf = io.BytesIO()
    doc = DocxDocument()
    doc.add_paragraph("第一段内容")
    doc.add_paragraph("第二段内容")
    doc.save(buf)
    text = parse_document(buf.getvalue(), "docx")
    assert "第一段内容" in text
    assert "第二段内容" in text


# ---- split_text ----


def test_split_text_long_text_produces_multiple_chunks() -> None:
    chunks = split_text(_long_text())
    assert len(chunks) >= 2
    settings = get_settings()
    for chunk in chunks:
        assert len(chunk) <= settings.chunk_size


def test_split_text_does_not_lose_content() -> None:
    chunks = split_text(_long_text())
    joined = "\n".join(chunks)
    # 允许 overlap 重复，但不允许丢句（首句与末句都必须出现）
    assert "这是第0句话" in joined
    assert "这是第199句话" in joined


def test_split_text_empty_is_no_chunks() -> None:
    assert [c for c in split_text("") if c.strip()] == []


def test_supported_exts_cover_parser_contract() -> None:
    assert {"txt", "md", "pdf", "docx"} == SUPPORTED_EXTS
