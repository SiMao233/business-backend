"""后台任务模块。

任务队列框架已确定为 **ARQ**（轻量、纯 asyncio、基于 Redis）。

- Worker 入口：`app/tasks/worker.py`（`uv run arq app.tasks.worker.WorkerSettings`）
- 当前任务：`run_agent_chat`（Agent 对话异步执行，由 app/modules/ai/ 入队）
- 预留：定时任务（缓存预热、数据同步、报表）、异步任务（文档切分索引、模型调用）
"""
