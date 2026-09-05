# HTML 审核页面

启动局域网服务：

```bash
python3 scripts/serve_review.py --host 0.0.0.0 --port 8765
```

然后在同一局域网设备访问：

```text
http://<本机局域网IP>:8765/web/review.html
```

页面默认自动读取正式 `100/20/30` 划分中的 30 份 test 文档第二审核队列。该队列包含 75 个文本块，已完成第二审核，实体和关系均为接受状态，并在每个文本块的审核元数据中保留自动关系校验发现的高风险关系。

正式划分和第二审核文件位于：

- `data/processed/reviewed/formal_split/second_review_queue.jsonl`
- `data/processed/reviewed/formal_split/jobs.jsonl`
- `data/processed/reviewed/formal_split/summary.json`

正式划分按文档级近重复簇隔离，目标为训练/验证/测试 `100/20/30`，不会覆盖当前 reviewed gold 的 pilot 划分。第二审核完成后，可导出 `reviewed_annotations.jsonl` 作为复核结果，后续再进行双审合并和正式测试冻结。

如需查看当前 reviewed gold，可手动导入：

- `data/processed/reviewed/gold/all.jsonl`
- `data/processed/reviewed/jobs.jsonl`
- `data/processed/reviewed/gold/record_index.jsonl`

历史统一审核队列文件为：

- `data/processed/experiments/annotation_pending_terra_all_review_queue.jsonl`
- `data/processed/experiments/annotation_pending_terra_all_jobs.jsonl`
- `data/processed/experiments/annotation_pending_terra_all_candidates.jsonl`
- `data/processed/experiments/annotation_pending_terra_batch2_jobs.jsonl`
- `data/processed/experiments/annotation_pending_terra_batch2_candidates.jsonl`

如果第二批文件不存在，则读取第一批中文教师审核批次：

- `data/processed/experiments/annotation_pending_terra_zh_review_queue.jsonl`
- `data/processed/experiments/annotation_pending_zh_jobs.jsonl`
- `data/processed/experiments/annotation_pending_terra_zh_candidates.jsonl`

如果当前批次文件不存在，页面会回退读取：

- `data/processed/reviewed/gold/all.jsonl`（优先加载当前已审核版本）
- `data/processed/preannotation/sub2api_terra_normalized.jsonl`
- `data/processed/preannotation/jobs.jsonl`
- `data/processed/preannotation/sub2api_terra_candidates.jsonl`（用于把重复文档的文本块精确映射到 `job_id`）

也可以通过“导入数据”加载 JSON/JSONL 文件。导入归一化标注时，建议同时导入候选映射 JSONL，以便逐个审核文本块。审核改动保存在当前浏览器的 `localStorage`，完成后使用“导出审核结果”生成 `reviewed_annotations.jsonl`。

外部图谱导入支持以下结构：

```json
{
  "nodes": [{"id": "n1", "label": "节点A", "type": "HAZARD"}],
  "edges": [{"source": "n1", "target": "n2", "type": "related_to"}]
}
```
