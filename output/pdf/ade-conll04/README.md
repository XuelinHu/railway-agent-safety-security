# 当前交付：ADE + CoNLL04

本目录是最新论文版本。上一级目录的旧文件名为历史稿，不再作为当前正文。

- [英文匿名全文（20页）](manuscript.pdf)
- [中文全文核对稿（15页）](manuscript-zh.pdf)
- [独立作者页](title-page.pdf)
- [可编辑匿名LaTeX包](review-source.zip)
- [独立图注](figure-captions.tex)
- [构建与审阅记录](FINAL_REVIEW.md)

正文包含一个主方法图和五个表（规模统计、标签计数、测试比较、两种子稳定性、
开发集消融）。DrawIO源文件及其PDF、PNG、SVG导出在`figures/`，已填充的投稿
Word附件在`submission-word/`。
统计CSV及可复现结果在`../../../paper/results/ade_conll04/`。

全文已围绕两套公开英文句级数据重写，不再混入旧语料、其他数据集或文档预算实验。
摘要减少参数和数字罗列；所有保留数据集上的真实基线、消融与限制仍如实报告。
本次没有启动新的模型训练或推理；种子2026与3407只读取已完成指标。
