# 三张论文图片的服务器端 AI 提示词

以下提示词面向能够读取服务器文件、运行 Python 并生成 SVG/Draw.io/PNG/PDF 的 AI 编程助手。使用前把 `<DATA_ROOT>` 替换为服务器上的真实数据目录。三张图都必须先检查数据文件和字段定义，再绘图；不得用常识补造样例。

## 提示词一：ADE 与 CoNLL04 转换为知识图谱后的真实样例

```text
你正在一个实体—关系抽取论文项目中工作。请读取 <DATA_ROOT> 下 ADE 与 CoNLL04 的真实 train/dev/test 文件，识别其实际 JSON/JSONL 字段、token、实体 span、实体类型、关系类型和关系端点方向，然后制作一张“从真实语料标注到知识图谱”的论文插图。

数据事实必须以文件内容为准，并执行以下约束：
1. 分成左右两个并列面板：左侧 ADE，右侧 CoNLL04。
2. 每个面板从真实数据中选择 1 个包含至少一条关系且句子长度适中的样例。图中逐字展示原句，实体文本必须与原始 token/span 完全一致；在原句中用颜色下划线或浅色底标出实体。
3. ADE 面板显示 Drug 与 Adverse-Effect 两类实体，以及真实 Adverse-Effect 关系。必须核对本数据文件的端点方向；若与当前项目约定一致，则箭头为“Adverse-Effect 实体 → Drug 实体”，不要擅自改成更直觉的反方向。例子必须来自文件，禁止使用 “aspirin–nausea” 等人工示例，除非它确实存在于数据中。
4. CoNLL04 面板显示真实出现的 Peop、Org、Loc、Other 实体，并从 Work_For、Kill、OrgBased_In、Live_In、Located_In 中展示该句真实包含的关系。优先选择包含 2 条或以上关系的句子，以便体现多实体、多关系图结构；若没有合适短句，再选择 1 条关系的真实句子。
5. 每个面板采用同样的三段结构：原始句子 → 实体/关系标注 → 小型知识图谱。知识图谱使用圆角节点，节点内写“实体原文 + 类型”，有向边上写关系标签；同一实体在同一句不同位置出现时，不得合并。
6. 在图底部用小字标注数据来源文件名、split、样例索引或 sentence/document ID，保证样例可追溯。不要显示服务器绝对路径。
7. 视觉风格：AAAI/SCI 论文插图，白底，扁平矢量，颜色克制，ADE 使用暖红/橙色，CoNLL04 使用蓝/青色；不使用阴影、3D、渐变、装饰性图标；文字必须清晰，整体横向 16:7。
8. 只画数据结构，不画模型、Transformer、训练过程或性能指标。

先输出一份“核验记录”，列出两个样例的原句、实体 span、实体类型、关系三元组、端点方向和来源位置；确认所有内容来自真实文件后，再生成可编辑 SVG 和 Draw.io，并导出 300 dpi PNG。最终告诉我产物路径。
```

## 提示词二：创新数据结构及其与传统方法的差别

```text
请阅读当前项目的论文、实现代码和 <DATA_ROOT> 中的 ADE/CoNLL04 数据格式，制作一张极简的“传统抽取 vs 本文创新”对比图。重点是本文输出的数据结构与证据门控，不要展开大语言模型或 Transformer 内部结构。

请先核对项目中 SOE、EAE、HRGE、EVGE、CFE、PGE 的真实定义，只在图中保留解释创新所必需的结构。推荐布局为左右对比：

左侧“Traditional extraction”：
- 输入句子 → 模型 → 实体与关系三元组；
- 输出仅表示 (head entity, relation, tail entity)，用一条虚线警示“关系可能缺少来源证据、端点或类型约束”；
- 整个左侧最多 3 个主视觉单元。

右侧“Provenance-gated extraction (ours)”：
- 输入句子及其精确字符 span + 训练集知识图谱上下文；
- 候选实体/关系进入两个极简门控：Entity gate = ANY（EAE 一致、source anchor、verified endpoint 任一满足）；Relation gate = ALL（引用有效、端点类型签名、显式状态、证据等约束全部满足）。门控条件应根据代码和论文核验，不能自行添加未实现条件；
- 输出为 Evidence-linked Assertion Graph。每个实体节点至少带 mention text、type、[start,end) span；每条关系边至少带 relation type、directed endpoints、evidence span/source ID、provenance、accept/reject reason 或 status 中项目实际存在的字段；
- 用一个真实 ADE 或 CoNLL04 样例作为右下角微型图谱，并标出关系边如何链接回原句证据。

需要突出且只突出三项创新：
1. 从“裸三元组”变成“带精确 span、来源和证据的断言图”；
2. 高召回候选与确定性验证分离；
3. 实体采用 ANY 接受逻辑，关系采用 ALL 合规逻辑，并在最终组合时过滤端点未存活的关系。

视觉限制：整张图最多 8 个外框；共享的 Qwen3-4B/QLoRA 只画成一个简洁模块并写名称，不画 token embedding、RMSNorm、GQA、MLP、LM Head、decoder block 等内部细节；SOE/CFE 等消融路径放入一行灰色脚注，不进入主流程。白底、横向、扁平矢量、无渐变无阴影，红色表示原始证据，紫色表示候选生成，绿色表示验证与最终图谱。文字简短，阅读顺序必须从左到右。

先给出你从代码中核验到的字段表与门控条件，再生成可编辑 SVG 和 Draw.io，并导出 PDF、300 dpi PNG。若论文描述与实现不一致，在绘图前明确列出冲突，不得自行选择一个版本。
```

## 提示词三：两套语料的真实输入形态与并列处理方式

```text
请在服务器项目中读取 <DATA_ROOT> 下 ADE 与 CoNLL04 的真实输入文件，以及项目将其转换为模型输入的预处理、prompt serialization、字符 span 重建和图上下文检索代码，制作一张“模型实际接收什么输入”的论文插图。

这两套数据是两个独立英文句级基准，不要称为双语或对齐的 parallel corpus。图中将它们画成两条并列输入流，并展示真实数据经过统一接口后的形态：

上半部分为两个真实样例卡片：
- ADE 卡片：展示 1 条真实原句、token 序列、Drug/Adverse-Effect 实体 span 和真实关系；
- CoNLL04 卡片：展示 1 条真实原句、token 序列、Peop/Org/Loc/Other 中实际出现的实体 span，以及 Work_For/Kill/OrgBased_In/Live_In/Located_In 中实际出现的关系；
- 样例必须从文件读取，保留原文大小写和标点，注明 split 与记录 ID/索引。

中间部分为一个共享的“Canonical sentence record”数据卡片，用接近 JSON 的简洁形式展示项目真实字段，例如：
{
  "text/tokens": ...,
  "entities": [{"text": ..., "type": ..., "start": ..., "end": ...}],
  "relations": [{"head/tail": ..., "type": ..., "direction": ...}],
  "source_id": ...
}
字段名必须以项目代码为准；如果原始数据使用 token span、模型输出使用字符 span，要用一条清晰箭头表示“published token span → exact source character [start,end)”，不能混写。

下半部分只展示模型输入的三种上下文视图：
1. SOE：source sentence + ontology，no KG context；
2. EAE：source sentence + exact-anchor hints；
3. HRGE：source sentence + bounded anchors/edges/relation patterns。
三种视图汇入同一个“Shared frozen Qwen3-4B + task-specific QLoRA”模块；模型模块只画一个框，不展开 Transformer。输出只写“structured candidate JSON”，不要在本图继续展开门控和最终 PGE。

图的核心信息是：两个数据集标签空间不同，但通过同一规范记录和三种上下文序列化方式进入共享模型架构；训练集 KG 只提供候选上下文，当前句子提供证据；不得使用验证集/测试集 gold 构造提示。若代码中存在 provenance isolation、同句排除或完全重合文本清空 graph hints 的规则，用一句短注释准确表达。

视觉风格：白底、横向 16:8、三层结构“真实语料 → 规范记录 → 三种输入视图”，总外框不超过 9 个；ADE 用暖红色，CoNLL04 用蓝色，共享记录用灰蓝色，模型输入用浅紫色。无装饰性图标、无复杂网络结构、无渐变和阴影。先输出两个真实样例及字段核验记录，再生成可编辑 SVG 和 Draw.io，并导出 300 dpi PNG。
```
