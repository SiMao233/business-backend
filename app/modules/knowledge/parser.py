"""文档解析与切分（RAG 索引前置处理）。

支持格式：txt / md（直读）、pdf（pypdf）、docx（python-docx）。
流程：文件字节 → 纯文本 → RecursiveCharacterTextSplitter 切分为块列表。
"""

import io

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import get_settings
from app.core.exceptions import ValidateError

# 支持解析的扩展名（不含点、小写）
SUPPORTED_EXTS = {"txt", "md", "pdf", "docx"}


def parse_document(data: bytes, ext: str | None) -> str:
    """解析文档字节为纯文本。ext 不含点、小写；不支持的格式抛 ValidateError。"""
    ext = (ext or "").lower()
    if ext not in SUPPORTED_EXTS:
        raise ValidateError(f"不支持解析的文件类型: {ext or '未知'}")
    if ext in ("txt", "md"):
        return data.decode("utf-8", errors="replace")
    if ext == "pdf":
        return _parse_pdf(data)
    return _parse_docx(data)


def _parse_pdf(data: bytes) -> str:
    """PDF → 纯文本（逐页提取，空页跳过）。"""
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _parse_docx(data: bytes) -> str:
    """docx → 纯文本（段落 + 表格单元格，表格行用 | 连接）。"""
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))
    return "\n".join(parts)


def split_text(text: str) -> list[str]:
    """按配置切分文本为块列表（chunk_size / chunk_overlap）。

    分隔符优先中文标点与换行，避免把语义切碎。
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_text(text)
