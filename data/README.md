# 数据目录

## `raw/raib`

保存从英国政府RAIB公开页面下载的官方事故调查报告PDF。RAIB页面标注适用英国 Open Government Licence，下载脚本只访问公开页面和官方 `assets.publishing.service.gov.uk` 文件，不绕过登录、验证码或付费墙。

## `raw/cn`

保存本地中文安全生产、消防、应急预案和演练资料。该目录作为本体、术语、控制措施和跨场景验证语料使用，不把通用企业模板视为铁路事故事实或正式监管依据。

## `catalog`

- `raib_manifest.csv`：报告页面、PDF地址、文件名和下载状态。
- `book_metadata.csv`：推荐电子书的书目信息和正版购买链接。出版社电子书受版权保护，仓库不保存其全文。
- `corpus_inventory.csv`：本地语料格式、哈希、重复关系和抽取状态，不保存正文。
- `pilot_set.csv`：首批中英文人工审核候选文档。

当前审核结果与知识图谱输出保存在被 Git 忽略的 `data/processed/reviewed/`：

- `gold/all.jsonl`：69 个文本块的当前 `gold v0.1.0` 标注。
- `gold/train.jsonl`、`gold/validation.jsonl`、`gold/test.jsonl`：按原始文档切分的训练、验证和测试标注。
- `knowledge_graph/concepts.jsonl`：规范化概念节点。
- `knowledge_graph/mentions.jsonl`：带文档、文本块、页码和原文证据的实体提及。
- `knowledge_graph/relations.jsonl`：带来源证据的关系实例。
- `split_manifest.jsonl`：文档级切分清单。
- `summary.json`：数量和版本摘要。

这些文件由以下命令可重复生成：

```bash
python3 scripts/promote_reviewed_annotations.py
```

## 预处理命令

```bash
python3 scripts/build_corpus.py
python3 scripts/select_pilot_set.py
python3 scripts/prepare_preannotation_jobs.py --teacher-model qwen3:14b
```

解析正文写入已忽略的 `data/processed/`。当前环境可可靠解析 PDF、DOCX 和 TXT。旧版 `.doc` 需要额外安装 `antiword`、`catdoc` 或 LibreOffice；工具缺失时文件会记录为失败，不会使用不可靠的字符串扫描代替正文解析。

## 采集命令

在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_public_corpus.ps1
```

测试少量报告：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_public_corpus.ps1 -MaxReports 20
```

由于原始报告会占用较大磁盘空间，`data/raw/`已被`.gitignore`忽略，不会自动提交到GitHub。提交前只提交manifest、元数据、代码和处理结果摘要。
