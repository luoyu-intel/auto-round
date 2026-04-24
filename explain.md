# outlier_profile.py 统计口径说明

这份文档说明 [outlier_profile.py](/home/yuluo/auto-round/outlier_profile.py) 当前版本的统计定义、JSON 输出字段、阈值统计口径，以及和旧版 commit `690b2826a72f3a78048f61bbf2daa97c970cb975` 的关系。

当前版本的核心变化只有一条：

先按最后一维切成 group，先在 group 内算统计量，再分别汇总成：

1. tensor 的 mean-group 统计
2. tensor 的 any-group 统计
3. 全局 group 统计

这样可以同时回答两类问题：

1. 一个 tensor 平均来看有多坏
2. 一个 tensor 里是否存在特别坏的 group

## 1. 基本切分方式

对任意 tensor，按最后一维按 `group_size` 切分。

例如：

- 原 shape: `[2, 128]`
- `group_size = 64`
- 重排后按 group 看，相当于 `[2, 2, 64]`

每个长度为 `group_size` 的子块都视为一个 group。后续所有底层统计都先在 group 内完成。

## 2. group 级基础统计

### 2.1 group max ratio

对每个 group，取绝对值最大的两个元素：

- `largest`
- `second_largest`

定义：

$$
	ext{group max ratio}(g) = \frac{\max_1(g)}{\max_2(g)}
$$

实现细节：

1. 如果 `second_largest > 0`，直接做除法。
2. 如果 `largest == 0` 且 `second_largest == 0`，定义 ratio 为 `1.0`。
3. 如果 `largest > 0` 且 `second_largest == 0`，定义 ratio 为 `inf`。

### 2.2 group paper K/r

当前版本的 group `paper K/r` 不再使用 `abs + quantile`。

现在采用的是符号敏感的端点规则。先在每个 group 里找绝对值最大的有符号端点 `K`：

1. 如果最大正数的绝对值不小于最小负数的绝对值，那么 `K` 取最大正数。
2. 否则 `K` 取最小负数。

然后按 `K` 的符号选择 `r`：

1. 如果 `K` 是正数，优先取最小负数作为 `r`。
2. 如果 `K` 是正数且 group 内没有负数，退化为 `second_largest_positive / 2`。
3. 如果 `K` 是负数，优先取最大正数作为 `r`。
4. 如果 `K` 是负数且 group 内没有正数，退化为 `second_smallest_negative / 2`。

定义：

$$
	ext{group paper K/r}(g) = \frac{|K(g)|}{|r(g)|}
$$

这样做的目的不是用分位数近似 inlier radius，而是直接逼近你说的 outlier 区间 `[r, K]` 或 `[K, r]`。

这套规则的效果是：

1. `group max_ratio = largest_abs / second_largest_abs` 仍然保留，用来描述绝对值尖刺。
2. `group paper K/r` 会利用符号信息区分“正向大 outlier”还是“负向大 outlier”。
3. 当 group 同时包含正负两侧值时，`paper K/r` 通常和 `max_ratio` 明显不同。
4. 只有在 group 基本单边时，才会退化到同号的次大值规则。

### 2.3 一个重要退化条件

现在 group `paper K/r` 已经没有按 percentile 的 rank-cap 退化问题。

`--kr-percentile` 仍然保留，但主要用于 `whole_tensor_paper_kr_ratio` 这条 legacy 统计，也就是整 tensor 展平后的旧口径比较值。

## 3. tensor 级两套统计

当前版本同时保留两种 tensor 统计语义。

### 3.1 mean-group 统计

这套统计回答的是：这个 tensor 的 group 平均有多坏。

对应字段：

- `mean_group_max_ratio`
- `mean_group_paper_kr_ratio`

定义：

$$
	ext{mean-group max ratio}(T) = \frac{1}{N}\sum_{g \in T} \text{group max ratio}(g)
$$

$$
	ext{mean-group paper K/r}(T) = \frac{1}{N}\sum_{g \in T} \text{group paper K/r}(g)
$$

这套统计适合判断：

1. 一个 tensor 是整体偏坏，还是整体比较平稳
2. 不同 tensor 的平均风险高低

### 3.2 any-group 统计

这套统计回答的是：这个 tensor 里有没有特别坏的 group。

对应字段：

- `any_group_max_ratio`
- `any_group_paper_kr_ratio`

定义：

$$
	ext{any-group max ratio}(T) = \max_{g \in T} \text{group max ratio}(g)
$$

$$
	ext{any-group paper K/r}(T) = \max_{g \in T} \text{group paper K/r}(g)
$$

这套统计本质上就是旧版的报警思路：

只要出现一个坏 group，这个 tensor 的 any-group 指标就会被抬高。

### 3.3 兼容别名

为了不立刻打断已有 JSON 消费逻辑，当前还保留了兼容字段：

- `max_ratio` = `mean_group_max_ratio`
- `max_group_ratio` = `any_group_max_ratio`
- `paper_kr_ratio` = `mean_group_paper_kr_ratio`
- `paper_kr_max_group_ratio` = `any_group_paper_kr_ratio`

新逻辑应优先读取显式字段，不建议继续依赖这些别名做长期接口。

## 4. 全局统计分三层

### 4.1 全局 group 统计

这部分是把所有 tensor 的所有 group 放在一起统计。

对应字段：

- `global_max_group_max_ratio`
- `global_mean_group_max_ratio`
- `global_max_group_paper_kr_ratio`
- `global_mean_group_paper_kr_ratio`

含义：

1. `global_max_*` 是所有 group 里的最坏 group
2. `global_mean_*` 是所有 group 的平均水平

如果你关心“全模型里坏 group 到底有多密集”，这组字段最直接。

### 4.2 全局 tensor mean-group 统计

这部分先对每个 tensor 算 mean-group，再在 tensor 层面聚合。

对应字段：

- `global_max_mean_group_max_ratio`
- `global_mean_mean_group_max_ratio`
- `global_max_mean_group_paper_kr_ratio`
- `global_mean_mean_group_paper_kr_ratio`

含义：

1. `global_max_*` 表示“平均最坏”的 tensor 有多坏
2. `global_mean_*` 表示所有 tensor 的平均 mean-group 水平

### 4.3 全局 tensor any-group 统计

这部分先对每个 tensor 算 any-group，再在 tensor 层面聚合。

对应字段：

- `global_max_any_group_max_ratio`
- `global_mean_any_group_max_ratio`
- `global_max_any_group_paper_kr_ratio`
- `global_mean_any_group_paper_kr_ratio`

含义：

1. `global_max_*` 是“最极端 tensor 的最极端 group”
2. `global_mean_*` 是“平均每个 tensor 的最坏 group 有多坏”

如果你想延续旧版“只要有一个坏 group 就报警”的阅读方式，主要看这一组。

## 5. threshold summary 怎么读

当前 threshold summary 分成两套：

1. `ratio_threshold_summary`
2. `paper_kr_threshold_summary`

每套又都分成两层：

1. `tensor_count`
2. `group_count`

### 5.1 `group_count`

这个最直接，就是对全模型所有 group 逐个计数：

- `greater_than_count`
- `greater_than_percentage`
- `less_than_count`
- `equal_count`

例如 `group_count.greater_than_percentage = 0.00129`，意思就是全部 group 中有 `0.129%` 超过阈值。

### 5.2 `tensor_count`

当前 tensor threshold 的规则不是看 mean-group 平均值，而是：

只要这个 tensor 里存在任意一个 group 超过阈值，就把这个 tensor 记为 `greater_than`。

也就是 any-group 语义。

所以：

1. tensor 占比高
2. group 占比低

这两件事并不矛盾。它表示的是：

很多 tensor 里都至少有一个坏 group，但坏 group 在所有 group 中的密度并不高。

这正是你前面怀疑的那类分布，现在文档里明确把它解释开了。

## 6. topk 输出怎么读

当前 summary 里和图里都把 topk 分成两份。

### 6.1 tensor topk

- `top_mean_group_tensors`
- `top_any_group_tensors`

前者用于找“整体平均偏坏”的 tensor。

后者用于找“存在尖刺 group”的 tensor。

### 6.2 layer topk

- `top_mean_group_layers`
- `top_any_group_layers`

这里的 layer 聚合是按 `extract_layer_name()` 和 module family 分桶后做的。

当前 layer 统计主要基于 paper K/r：

- `mean_paper_kr_ratio`
- `max_paper_kr_ratio`
- `mean_any_group_paper_kr_ratio`
- `max_any_group_paper_kr_ratio`

## 7. 四张图分别是什么

运行后当前会输出四张图：

1. `figure_output`
2. `any_group_figure_output`
3. `layer_figure_output`
4. `any_group_layer_figure_output`

对应关系如下：

| 字段 | 含义 |
|---|---|
| `figure_output` | tensor 级 mean-group paper K/r 图 |
| `any_group_figure_output` | tensor 级 any-group paper K/r 图 |
| `layer_figure_output` | layer 级 mean-group paper K/r 图 |
| `any_group_layer_figure_output` | layer 级 any-group paper K/r 图 |

阅读建议：

1. 先看 `figure_output`，判断哪些 tensor 是整体偏坏。
2. 再看 `any_group_figure_output`，判断哪些 tensor 只是局部有尖刺。
3. 再看两个 layer 图，判断问题更集中在哪类层和模块族上。

## 8. 与旧版 commit `690b2826...` 的对照

下面是当前版本和 commit `690b2826a72f3a78048f61bbf2daa97c970cb975` 的主要差异。

| 项目 | 旧 commit `690b2826...` | 当前版本 |
|---|---|---|
| group 切分 | 按最后一维切 group | 按最后一维切 group |
| group `max_ratio` | 逐 group 计算 `largest / second_largest` | 相同 |
| tensor `max_ratio` | 取所有 group 的最大值 | 取所有 group 的平均值，并额外保留 any-group 最大值 |
| tensor `paper_kr_ratio` | 整个 tensor 展平后计算一次 | 逐 group 计算后再聚合 |
| group `paper K/r` | 没有显式统计 | 显式逐 group 统计 |
| threshold tensor 规则 | tensor 自身指标过阈值 | 现在按 any-group 规则计数 |
| threshold group 规则 | 无 | 对全部 group 逐个统计 |
| 能回答的问题 | 找最坏 tensor | 同时看平均风险和尖刺风险 |

### 8.1 旧版更像什么

旧版更接近：

“这个 tensor 里只要有一个极端坏 group，我就把它视作坏 tensor。”

### 8.2 当前版更像什么

当前版拆成两条并行视角：

1. mean-group: 平均风险
2. any-group: 报警风险

所以当前版不是推翻旧版，而是把旧版语义显式保留下来，同时补上平均视角和 group 密度视角。

## 9. 建议优先关注哪些字段

如果你的目标是复现旧版直觉，优先看：

1. `any_group_max_ratio`
2. `any_group_paper_kr_ratio`
3. `global_max_any_group_max_ratio`
4. `global_max_any_group_paper_kr_ratio`
5. `top_any_group_tensors`
6. `top_any_group_layers`
7. threshold summary 里的 `tensor_count`

如果你的目标是看模型整体风险分布，优先看：

1. `mean_group_max_ratio`
2. `mean_group_paper_kr_ratio`
3. `global_mean_group_max_ratio`
4. `global_mean_group_paper_kr_ratio`
5. `top_mean_group_tensors`
6. `top_mean_group_layers`
7. threshold summary 里的 `group_count`

## 10. 一句话总结

当前版本把 outlier profiling 拆成了三层视角：

1. group 本身有多坏
2. tensor 平均有多坏
3. tensor 是否至少包含一个坏 group

其中：

- mean-group 用来描述平均风险
- any-group 用来保留旧版报警语义
- global group 统计用来描述坏 group 的真实密度