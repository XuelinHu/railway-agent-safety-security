# 数据目录

## `raw/raib`

保存从英国政府RAIB公开页面下载的官方事故调查报告PDF。RAIB页面标注适用英国 Open Government Licence，下载脚本只访问公开页面和官方 `assets.publishing.service.gov.uk` 文件，不绕过登录、验证码或付费墙。

## `catalog`

- `raib_manifest.csv`：报告页面、PDF地址、文件名和下载状态。
- `book_metadata.csv`：推荐电子书的书目信息和正版购买链接。出版社电子书受版权保护，仓库不保存其全文。

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
