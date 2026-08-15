# Railway Agent for Safety and Security：研究需求与语料方案

## 1. 研究目标

面向 Special Issue “Agent for Safety and Security: Complex Network Modeling, Substructure Analysis, Risk Discovery, and Resilience Assessment”，拟研究铁路复杂系统中的低概率风险、事故前兆、级联风险和网络韧性。

推荐的首篇英文论文问题是：

> How can a multi-agent LLM system discover latent risk substructures from railway accident and near-miss reports, verify them against safety regulations, and assess their cascading impact on railway-network resilience?

暂定英文题目：

> Discovering Latent Risk Substructures from Railway Accident Reports Using Large Language Model Agents

## 2. 语料不应只由电子书组成

电子书适合用于：

- 学习铁路安全、风险管理和韧性评估理论；
- 建立铁路安全术语表和风险本体；
- 设计实体类型、关系类型和标注规范；
- 支撑论文的背景和方法论引用。

真正的实验语料应以事故调查报告、险情报告、官方规章标准和运行/基础设施数据为主。这样才有可验证的事实、时间线、因果关系和安全建议。

## 3. 推荐语料结构

### 3.1 核心语料：事故与险情报告

建议优先收集英文报告，构建 300--600 份事故调查报告和 100--300 份 near-miss / safety-digest 报告。

推荐字段：

| 字段 | 内容 |
|---|---|
| event_id | 事故或险情唯一编号 |
| date/location | 时间和地点 |
| railway_type | heavy rail、metro、light rail 等 |
| event_type | collision、derailment、fire、SPAD、level-crossing 等 |
| actors | driver、dispatcher、maintainer、passenger 等 |
| assets | track、signal、switch、train、power、communication 等 |
| precursors | 事故前兆、异常、隐患和near miss |
| causal_factors | 技术、人因、组织、环境和管理因素 |
| controls | 已有防护、屏障、规章和应急措施 |
| recommendations | 调查机构提出的安全建议 |
| evidence | 报告页码、章节或原文证据 |

首选来源：

- [英国RAIB调查报告库](https://www.gov.uk/government/collections/catalogue-of-investigation-reports-and-bulletins)：报告通常包含事实、证据、调查、原因、结论和建议，适合构建事件因果图。
- [美国FRA Safety Data](https://railroads.dot.gov/safety-data)：提供铁路事故、事件、伤亡、道口和运营相关数据。
- [美国FRA Accident Data and Investigations](https://railroads.dot.gov/railroad-safety/accident-data-reporting-and-investigations-0)：补充事故数据定义、报告和调查资料。

### 3.2 规范语料：安全管理和风险评估文件

用于回答“某个风险是否已被规范覆盖”“已有何种控制措施”“管理系统是否存在缺口”。

推荐加入：

- [ERA Common Safety Method for Risk Evaluation and Assessment](https://www.era.europa.eu/domains/common-safety-methods/risk-evaluation-assessment-csm_en)
- [ERA Safety Management System Requirements](https://www.era.europa.eu/domains/common-safety-methods/safety-management-system-requirements-csm_en)
- Network Rail Health and Safety Management System
- FRA railway safety regulations and guidance
- RSSB公开的铁路安全指导材料

### 3.3 网络和运行语料

用于把文本中的风险子结构映射为铁路复杂网络：

- 车站、线路、道岔和控制中心拓扑；
- 列车运行、晚点、中断和恢复记录；
- 设备故障、维修和检修记录；
- 极端天气、洪水、滑坡、高温和暴雪记录；
- 道口、危险货物运输和应急资源位置。

### 3.4 辅助语料：新闻和公众信息

新闻可以用于早期事件发现和信息传播分析，但不应作为事故原因的 ground truth。新闻存在重复、时间线不完整和责任归因未确认等问题。

## 4. 国内铁路语料来源

### 4.1 国家铁路局：第一优先级

国家铁路局是国内最重要的公开权威来源。

- [国家铁路局监管履职](https://www.nra.gov.cn/jglz/)：包含法规制度、标准规范、安全监管和统计信息。
- [现行铁路行业标准和铁路国家标准目录](https://zwfw.nra.gov.cn/art/2024/7/31/art_182_7681.html)：截至2024年7月31日，目录列出铁路行业标准和铁路国家标准，可按装备技术、工程建设、运输服务查询。
- [铁路技术标准信息查询](https://app.gjzwfw.gov.cn/jmopen/webapp/html5/tljsbzxxfwpt/index.html)：查询标准编号、标准名称、状态、发布日期和实施日期。
- [国家铁路局法规制度](https://www.nra.gov.cn/jglz/fgzd/)：包括《铁路安全管理条例》《铁路交通事故应急救援和调查处理条例》及相关部门规章、规范性文件。
- [2024年铁道统计公报](https://source.nra.gov.cn/xwzx/zlzx/hytj/202506/t20250606_348988.shtml)：可用于年度铁路运量、线路、设备和安全指标背景分析。

### 4.2 国内事故和隐患语料

建议重点收集：

- 国家铁路局公开的铁路交通事故调查报告；
- 重大铁路交通事故和典型事故案例；
- 铁路安全监管执法、隐患排查和整改信息；
- 铁路沿线环境、道口、施工、危险货物和自然灾害风险信息。

特别值得纳入的是国家铁路局发布的《铁路交通重大事故隐患判定标准》及政策解读。该标准将隐患分为主要行车设备设施、铁路运输生产、铁路沿线环境、安全管理、灾害防范与应急处置五类，天然适合设计风险本体和分类标签。

- [铁路交通重大事故隐患判定标准](https://source.nra.gov.cn/xxgk/gkml/ztjg/gfzd/gfxw/zuti/jgzf/202606/t20260612_351429.shtml)
- [判定标准政策解读](https://source.nra.gov.cn/xxgk/gkml/ztjg/gfzd/zcjd/202606/t20260612_351430.shtml)

### 4.3 交通运输部和国家标准平台

- [交通运输部政府信息公开](https://xxgk.mot.gov.cn/)：补充铁路建设工程安全生产、应急管理和运输监管文件。
- [国家标准全文公开系统](https://openstd.samr.gov.cn/bzgk/std)：查询铁路工程、安全生产、应急管理和信息安全相关国家标准。
- [中国铁道出版社](https://www.tdpress.com/)：适合购买铁路安全管理、行车组织、信号、工务、电务和应急处置类正版教材与标准解读资料。

### 4.4 国内数据的使用建议

国内公开事故调查报告的数量和格式可能不如RAIB统一，因此建议采用“双语双来源”设计：

1. 用RAIB/FRA英文报告作为主要训练、测试和可复现实验语料；
2. 用国家铁路局公开文件作为中文跨制度验证集；
3. 不直接把中文报告机器翻译后混入英文主测试集；
4. 单独报告中文语料的数量、类型、时间范围、翻译方法和人工校验比例。

国内语料可以形成一个有价值的扩展实验：比较中国铁路安全隐患分类与欧洲CSM风险分类之间的差异，分析不同监管制度下的风险子结构迁移能力。

## 5. 推荐购买的正版电子书

建议先购买前三本，其他书按研究方向补充。

### 第一优先级

1. **Managing Risks in the Railway System: A Practice-Oriented Guide**。适合风险识别、风险分析、风险控制、风险沟通和复杂铁路系统风险管理。
   - [Springer电子书页面](https://link.springer.com/book/10.1007/978-3-030-66266-0)

2. **Railway Infrastructure Security**。适合关键基础设施保护、危机管理、网络安全和铁路韧性。
   - [Springer电子书页面](https://link.springer.com/book/10.1007/978-3-319-04426-2)

3. **Railway Safety Management: Systems, Practices, and Emerging Trends**。适合SMS、维护管理、安全文化、风险管理和预测性维护。
   - [Routledge电子书页面](https://www.routledge.com/Railway-Safety-Management-Systems-Practices-and-Emerging-Trends/Chruzik-Grabon-Chalupczak/p/book/9781041111870)

### 按专题补充

4. **Rail Human Factors: Supporting the Integrated Railway**。适合人因、疲劳、工作负荷、态势感知、控制中心和安全文化。
   - [Google Play电子书页面](https://play.google.com/store/books/details/J_Wilson_Rail_Human_Factors?id=qy4rDwAAQBAJ)

5. **Handbook of RAMS in Railway Systems: Theory and Practice**。适合可靠性、可用性、可维护性、安全性和设备故障分析。
   - [Google Play电子书页面](https://play.google.com/store/books/details/Qamar_Mahboob_Handbook_of_RAMS_in_Railway_Systems?id=QGNRDwAAQBAJ)

6. **Active Safety Methodologies of Rail Transportation**。适合主动安全、状态识别、故障预测和铁路网络级风险评估。
   - [Springer电子书页面](https://link.springer.com/book/10.1007/978-981-13-2260-0)

7. **Railway Security: Protecting Against Manmade and Natural Disasters**。适合人为灾害、自然灾害、危险品运输和应急响应。
   - [Routledge电子书页面](https://www.routledge.com/Railway-Security-Protecting-Against-Manmade-and-Natural-Disasters/Young-Gordon-Plant/p/book/9781420080643)

## 6. Agent系统设计

### Extraction Agent

抽取事故事件、时间、地点、设备、人员、环境、控制措施、失效和安全建议，并保存原文证据位置。

### Causal Analysis Agent

重建事故时间线，识别 precursor、barrier failure、causal factor、latent condition 和 cascading path。

### Regulation Verification Agent

检索ERA、国家铁路局、FRA或Network Rail规范，判断风险是否已有控制要求，以及事故中是否存在合规缺口。

### Resilience Assessment Agent

将风险子结构映射到铁路网络，计算关键节点、脆弱边、级联传播路径、服务损失和恢复能力。

所有Agent输出必须包含：结论、证据原文、来源、置信度、推理链摘要和人工复核状态。

## 7. 建议评价指标

- 实体和关系抽取：Precision、Recall、F1；
- 事故因果链：事件顺序准确率、因果关系F1；
- 规范匹配：evidence-grounded accuracy；
- 低概率风险发现：专家标注的发现率、误报率和新颖性；
- 子结构识别：与人工标注子图的graph edit distance或graph F1；
- 级联风险：关键节点识别的Precision@K和排序相关性；
- 解释性：证据覆盖率、来源可追溯率和专家评分；
- 韧性评估：网络性能下降、恢复时间、服务恢复率和鲁棒性曲线。

## 8. 版权、伦理与可复现性

- 只使用公开、合法取得或获得授权的数据；
- 电子书只用于个人研究和方法参考，不上传到GitHub；
- GitHub仓库只保存元数据、下载脚本、处理代码、标签模板和公开数据链接；
- 不提交事故报告的完整受版权保护副本，除非其明确允许再分发；
- 对事故受害者和工作人员信息进行必要的脱敏；
- 记录数据版本、下载日期、处理脚本版本、模型版本和人工复核结果。

## 9. 首阶段工作清单

- [ ] 建立国内外数据源清单和metadata表
- [ ] 下载并整理20份RAIB报告和20份国内公开安全文件作为试验集
- [ ] 设计中英文铁路风险本体
- [ ] 制定实体、关系、因果链和安全建议标注规范
- [ ] 完成一个可追溯的报告解析基线
- [ ] 实现四类Agent的最小可运行版本
- [ ] 与纯LLM、RAG和传统信息抽取方法进行对比
- [ ] 完成英文论文大纲、实验协议和数据声明

## 10. 目标投稿时间

Special Issue给出的投稿截止日期为 **2027年2月28日**。建议在此之前完成至少一轮专家复核、消融实验和跨来源验证。
