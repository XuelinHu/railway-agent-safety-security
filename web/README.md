# HTML 审核页面

启动局域网服务：

```bash
python3 scripts/serve_review.py --host 0.0.0.0 --port 8765
```

然后在同一局域网设备访问：

```text
http://<本机局域网IP>:8765/web/review.html
```

页面会自动读取：

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
