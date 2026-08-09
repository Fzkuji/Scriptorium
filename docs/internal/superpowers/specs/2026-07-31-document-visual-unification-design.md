# NativeMem 对外文档与方法总览图统一设计

> 状态：已确认设计
> 更新：2026-07-31

## 1. 目标

统一 NativeMem 当前四个对外 HTML 页面的页面结构与视觉规则，并重画已经不能完整表达当前方法的方法总览图。只调整展示层，不改论文分类、方法定义、实验数据、章节顺序或页面链接语义。

## 2. 范围

本次修改覆盖：

- `Model-Aligned-Wiki.html`：项目总览；
- `docs/survey.html`：Related Work；
- `docs/method/index.html`：当前 Method；
- `docs/experiment-plan.html`：实验计划与结果；
- `docs/assets/document.css`：新增的共享页面样式；
- `code/figures/nativemem_overview.svg`：可编辑的方法总览图源文件；
- `code/figures/nativemem_overview.png`：用于兼容现有引用的导出图。

以下内容不在本次范围内：

- `docs/method/report/**`、`docs/method/versions/**` 和 `docs/archive/**` 等历史记录；
- `docs/prompts_collection/**` 中逐字保存的 prompt；
- 运行日志、计划、benchmark fixture 和生成的预览页面；
- 独立的 `paper/` Git 仓库；
- Markdown 文档的批量元数据或标题格式改写。

根目录的 `experiment-plan.html` 是指向 `docs/experiment-plan.html` 的兼容链接，不单独修改。

## 3. 页面架构

四个页面继续采用静态 HTML，不引入站点生成器、前端框架或新的构建步骤。公共样式集中到 `docs/assets/document.css`，各页面只保留与自身组件直接相关的少量 CSS 和 JavaScript。

统一页面结构为：

1. 左侧固定目录；
2. 右侧正文；
3. 正文顶部 breadcrumb、页面标题、说明和更新时间；
4. 统一的章节标题、表格、代码块、提示框和图注；
5. 阅读进度条与当前章节高亮。

页面职责保持不变。Related Work 不加入 NativeMem 方法描述，Method 不吸收 Related Work 的系统分类，Experiments 保留现有 tabs、结果状态和节点移动逻辑。

## 4. 视觉规范

采用“统一研究文档版式”：

- 桌面端目录宽度固定为 `256px`；
- 正文最大宽度约为 `1080px`，页面内居中；
- 桌面端正文左右内边距为 `48px`，窄屏逐级收缩；
- `Noto Sans SC`、`Inter` 和 `JetBrains Mono` 分别用于中文正文、拉丁字符和代码；
- 主强调色统一为 `#4263eb`；
- 页面标题、说明和更新时间采用相同的字号、行高和间距；
- `h2` 使用统一的蓝色下边线，`h3` 不重复使用章节分隔线；
- 表格均置于可横向滚动容器中，避免窄屏截断；
- callout 使用统一容器结构，仅通过左边线颜色表达语义；
- 代码、链接、图注和 breadcrumb 使用同一套颜色与圆角规则。

页面固定使用浅色主题。本次不增加主题切换，不保留无法实际启用的重复深色规则。

在小于 `1024px` 的视口中隐藏固定侧栏，显示紧凑的页面导航入口；表格和图保持自身横向滚动能力。交互元素保留键盘焦点样式，并遵循 `prefers-reduced-motion`。

## 5. 方法总览图

采用“多视图状态中心”布局，画布按宽幅论文图设计。Source Memory 位于图中央，其他视图、Agent 操作和查询流程围绕该状态组织：

1. **Agent Operations** 位于左侧，包括 Memory Writer、Memory Manager 和 Query Navigator；
2. **Multi-View Text Memory** 位于中央，以 Source Memory 为中心，Topical View、Temporal View、Recent Memory 和 Core Memory 分布在四周；
3. **Hyperlink Relations and Source References** 在中央状态区内连接各视图，并指向可核验的 Source Memory；
4. **Query-Time Retrieval** 位于右侧，包括 `grep`、BM25、Embedding、结构化文件访问、selected evidence 和 evidence-grounded answer；
5. **Three-Level Memory Management** 作为 Memory Manager 的明确说明，包括 incremental writing、retrieval-triggered local reorganization 和 daily global management。

图中明确表达：

- 文本文件是权威状态；
- Source Memory 追加写入并保存完整原始交互；
- Topical View、Temporal View、Recent Memory 和 Core Memory 是面向不同访问需求的文件式状态；
- `grep`、BM25 和 Embedding 是由 Agent 自主选择的并列检索接口；
- 文件、timeline、links 和 source resolution 属于结构化访问；
- Runtime 负责路径、日期、source references、链接、预算和提交一致性；
- BM25 与 Embedding 索引可从文本文件重建，不是权威事实来源。

图中不加入具体的默认轮数、token 上限、benchmark 配置或尚未完成的实现状态。这些信息继续由 Method 和 Experiments 页面说明。

视觉颜色与页面保持一致：

- 蓝色：查询与控制流；
- 橙色：验证、提交与 Source Memory；
- 绿色：文件式记忆状态；
- 紫色：LLM 决策和周期管理；
- 灰色：工具接口与辅助结构。

SVG 是可编辑源文件；PNG 从同一 SVG 导出，不单独维护第二套图形内容。页面优先引用 SVG，PNG 保留给不支持 SVG 的外部使用场景。

## 6. 兼容性与验证

实现完成后执行以下检查：

- 四个页面均包含 `DOCTYPE`、正确的 `lang`、charset 和 viewport；
- 页面 ID 唯一，目录锚点、页面内链接和本地相对链接均有效；
- Overview、Related Work、Method 和 Experiments 的正文起点、目录宽度、标题尺度与页头间距一致；
- 1440px 桌面视口与 390px 移动视口下无非预期页面级横向溢出；
- 所有表格在窄屏可横向滚动；
- Experiments 的 tabs、动态节点移动和结果状态交互保持可用；
- SVG 可直接打开，文字不越界，缩放到页面正文宽度后仍可辨认；
- PNG 与 SVG 的内容和宽高比例一致；
- `git diff --check` 通过。

## 7. 实现边界

不增加新的 JavaScript 依赖，不引入模板系统，不重写页面正文。共享样式只统一重复的页面规则；实验页和图表所需的专用样式继续留在各自页面中。历史页面保持原有视觉形式，以免改变历史记录的展示语境。
