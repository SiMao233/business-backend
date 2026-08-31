"""AI 能力 Repository 层：数据访问。

第一版对话闭环不落库，本层暂不实现具体数据访问。
后续新增会话/运行记录表时，在此基于 BaseRepository 实现。
"""

from app.common.repository import BaseRepository


class AiRepository(BaseRepository):
    """AI 数据访问仓库（占位，待会话/运行记录表实现后指定泛型参数）。"""
