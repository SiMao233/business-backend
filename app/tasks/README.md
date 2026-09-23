# 后台任务模块（ARQ）

任务队列已落地：**ARQ**（纯 asyncio，Broker 复用项目已有的 Redis）。
入口为 `app/tasks/worker.py` 的 `WorkerSettings`。

## 任务清单

| 任务 | 入队方 | 说明 |
| --- | --- | --- |
| `process_document` | `knowledge/service.py`（上传 / 重试接口） | 解析 → 结构感知切分 → embedding → Qdrant / MySQL 双写 |
| `run_agent_chat` | `ai/chat/service.py` | 后台执行 Agent 对话 |

## 启动

```bash
uv run arq app.tasks.worker.WorkerSettings
```

`WorkerSettings`：`max_jobs=10`、`job_timeout=300`，Redis 连接参数取自 `app/core/config.py`。
容器编排中由 `worker` 服务启动（见 `docker-compose.yml`），与 app 共用同一镜像。

## 约定与坑（改这块前必读）

1. **入队一律不传 `job_id`**：ARQ 的重复判定是「`job_key` 或 `result_key` 存在」，而 `result_key`
   的 TTL 是 `WorkerSettings.keep_result`（默认 **1 小时**）→ 传固定 `job_id` 会让「任务结束后 1 小时内
   的同 ID 入队」被**静默丢弃**（返回 `None`，不报错），表现为「重试接口调了但任务没跑」。
   并发保护改由任务内的 **CAS 抢占**（`DocumentRepository.mark_parsing()`）承担。
2. **文档索引是三阶段生命周期**：抢占 → 只读计算（解析 / 切分 / embedding，全程不碰旧索引）
   → 替换（先清后写）。阶段 2 失败时旧索引完好，文档标 `failed`，重试即可。
3. **取消语义**：ARQ `job_timeout` 取消任务时抛 `CancelledError`（属 `BaseException`，
   `except Exception` 捕不到）→ 必须用 `except BaseException` 捕获，并在
   `anyio.CancelScope(shield=True)` 内完成状态落库后再重新抛出，否则文档会永久卡在 `parsing`。
4. **新增任务**：在 `worker.py` 实现协程函数 → 加入 `WorkerSettings.functions` → 通过
   `enqueue_job(...)` 入队；业务模块请**延迟导入**，避免 worker 启动时的循环依赖。
5. **僵尸任务**：worker 被 SIGKILL 等无法执行兜底逻辑的场景，由 `_is_stale_parsing()`
   按 `INDEX_STALE_MINUTES`（默认 15 分钟）判定并允许重试。

## 规划

- **定时任务**：缓存预热、数据同步、报表生成等（`WorkerSettings.cron_jobs` 已预留）
- **更多异步任务**：批量重索引、评测任务等
