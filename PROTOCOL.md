# 缺失模式不匹配对时间序列分类方法选择的影响

## 一项受控实证研究

**English title:**
**The Effect of Missingness-Pattern Mismatch on Method Selection for Time-Series Classification: A Controlled Empirical Study**

**Protocol version:** v1.5
**Project status:** Development / Protocol updated
**Project type:** Controlled empirical study
**Primary task:** Univariate time-series classification

---

## 1. 研究问题

本研究关注一个具体的模型选择问题：

> 在缺失比例相同的情况下，如果使用一种缺失模式的验证数据来选择分类方法，而最终测试数据具有另一种缺失模式，那么这种缺失模式不匹配是否会影响最终选择的分类方法，以及影响最终分类性能的程度有多大？

研究重点放在 **validation missingness pattern 与 test missingness pattern 不一致** 的情况。

核心比较为：

* 使用 Point missingness 验证数据选择方法，再在 Point missingness 测试数据上评估；
* 使用 Point missingness 验证数据选择方法，再在 Block missingness 测试数据上评估；
* 使用 Block missingness 验证数据选择方法，再在 Point missingness 测试数据上评估；
* 使用 Block missingness 验证数据选择方法，再在 Block missingness 测试数据上评估。

通过这种设计，将匹配条件与不匹配条件放在相同的数据、相同的候选方法和相同的测试流程下进行比较。

---

## 2. 研究对象

本研究针对：

* 单变量（univariate）时间序列；
* 等长度（equal-length）时间序列；
* 监督式时间序列分类（time-series classification）；
* 公开可获得的数据集；
* 以 UCR/UEA 风格数据集为主要数据来源。

### 2.1 主实验数据集

主实验固定使用 aeon 所列 UCR2015 archive 中的 64 个单变量、等长数据集：

`Adiac`, `ArrowHead`, `Beef`, `BeetleFly`, `BirdChicken`, `Car`, `CBF`,
`Coffee`, `Computers`, `CricketX`, `CricketY`, `CricketZ`, `FaceAll`,
`DistalPhalanxOutlineCorrect`,
`DistalPhalanxOutlineAgeGroup`, `DistalPhalanxTW`, `Earthquakes`, `ECG200`,
`ECG5000`, `ECGFiveDays`, `FaceFour`, `FacesUCR`, `Fish`, `GunPoint`,
`Ham`, `Haptics`, `CinCECGTorso`, `InsectWingbeatSound`, `ItalyPowerDemand`,
`LargeKitchenAppliances`, `Lightning2`, `Lightning7`, `Meat`,
`MedicalImages`, `MiddlePhalanxOutlineCorrect`,
`MiddlePhalanxOutlineAgeGroup`, `MiddlePhalanxTW`, `MoteStrain`, `OliveOil`,
`OSULeaf`, `PhalangesOutlinesCorrect`, `Plane`,
`ProximalPhalanxOutlineCorrect`, `ProximalPhalanxOutlineAgeGroup`,
`Mallat`, `RefrigerationDevices`, `ScreenType`, `ShapeletSim`,
`ShapesAll`, `SmallKitchenAppliances`, `SonyAIBORobotSurface1`,
`SonyAIBORobotSurface2`, `Strawberry`, `SwedishLeaf`, `Symbols`,
`SyntheticControl`, `ToeSegmentation1`, `ToeSegmentation2`, `Trace`,
`TwoLeadECG`, `Wine`, `WordSynonyms`, `Worms`, `WormsTwoClass`.

该清单在运行任何结果分析前固定。为使 1NN-DTW 在本地可执行，UCR2015 中
21 个样本量与序列长度组合特别大的数据集被基于计算资源预先排除；排除不参考
分类性能、缺失模式或实验结果。

在运行正式结果前的加载预检中，发现 7 个 aeon 本地缓存文件在最后一条序列处被截断，
导致 aeon 报出维度不一致或错误的类别计数。这些目录被逐个删除并由 aeon 重新下载
官方数据；`CricketZ`、`MiddlePhalanxOutlineAgeGroup` 和 `MiddlePhalanxTW` 在
修复后通过完整加载与分层划分预检。

重新下载后，`DiatomSizeReduction` 与 `FiftyWords` 的官方训练数据仍各有一个类别
只有 1 条序列，不能执行本研究固定的 stratified train/validation split；`Herring`
和 `ProximalPhalanxTW` 仍无法被 aeon 1.6.0 读取。为保留预先规定的 64 个数据集
规模，v1.4 在正式结果产生前以同属 UCR2015、单变量、等长并通过加载与分层预检的
`FaceAll`、`FacesUCR`、`CinCECGTorso`、`Mallat` 分别替换它们。所有替换均不参考
任何分类性能、缺失模式或实验结果。

每个样本表示为：

$$
x=(x_1,x_2,\ldots,x_T)
$$

其中：

* \(T\) 为时间序列长度；
* \(x_t\) 为时间点 \(t\) 的观测值；
* 每个样本具有一个分类标签。

本研究暂不讨论：

* 多变量时间序列；
* MAR/MNAR 等更复杂的缺失机制；
* 医疗领域专门适配；
* foundation models；
* Transformer 类大型模型；
* 新型分类器设计；
* 新型缺失度量；
* 新型插补算法。

---

## 3. 缺失模式

研究使用两种主要缺失模式。

### 3.1 Point missingness

Point missingness 表示随机散点缺失。

对于时间序列长度 \(T\)，指定缺失比例 \(r\)，缺失数量定义为：

$$
k=\lfloor rT\rfloor
$$

从 \(T\) 个时间位置中均匀随机抽取 \(k\) 个不同位置，使这些位置成为缺失值。

每个位置最多被选中一次。

因此 Point missingness 的缺失位置彼此通常不连续。

记该缺失模式为：

$$
P
$$

---

## 4. Block missingness

Block missingness 表示连续的一段时间区间发生缺失。

主实验采用 **circular contiguous block**：

1. 根据缺失比例计算

   $$
   k=\lfloor rT\rfloor
   $$
2. 在时间轴上随机选择一个起始位置；
3. 从该位置开始连续选取 \(k\) 个时间点；
4. 时间轴按循环方式处理，因此区间允许跨越序列末端重新回到开头。

例如长度为 \(T=20\)，若 \(k=5\)，某个 block 可以表现为：

$$
(18,19,20,1,2)
$$

这种设计使每个时间位置在重复随机放置中具有相同的边际被遮蔽机会。

记该缺失模式为：

$$
B
$$

### 4.1 关于 circular block 的说明

Circular block 用于控制实验中的位置暴露差异。

本研究不会将 circular block 解释为现实世界中某一种特定传感器故障机制。

它主要用于构造一个具有明显局部连续结构、同时避免固定边界效应的受控实验条件。

Circularity 仅属于 masking construction；它不意味着底层时间序列具有周期性。

---

## 5. Supplementary Linear Block

除主实验使用的 circular block 外，再进行普通的线性连续 block masking 作为补充实验。

在线性 block 中：

* 缺失区间必须完全位于 \(1,\ldots,T\) 内；
* 不允许跨越时间序列首尾连接。

该条件用于检查主要观察结果是否依赖于 circular block 的具体构造。

Linear block 不改变主实验问题，也不成为新的主要研究变量；该分析仅用于检验主要结果对 block masking construction 的敏感性。

### 5.1 Supplementary experiment design

Linear-block robustness analysis 使用与主实验相同的 64 个数据集、5 个 random seeds、6 个 nominal missingness rates、3 个候选分类器、train/validation/test procedure、linear-interpolation procedure、model-selection rule 和 evaluation metrics。

唯一的改变是 block missingness 的构造方式：补充实验使用 **non-wrapping linear contiguous blocks**，而不是主实验中的 circular contiguous blocks。

由于 Point→Point 条件不涉及 block missingness，其主实验结果直接复用。需要重新运行的条件为：

* Point→Block（PB）；
* Block→Point（BP）；
* Block→Block（BB）。

因此补充实验新增：

$$
64\times5\times6\times3\times3=17,280
$$

次 classifier evaluations。

对于相同的 dataset × seed × missingness rate：

* BP 与 BB 使用同一个 linear-block validation mask；
* PB 与 BB 使用同一个 target linear-block test mask；
* Point missingness 的 mask generation 规则保持不变。

该配对结构用于保持与主实验相同的 target-paired comparison framework。

Linear-block robustness analysis 单独分析，不与 circular-block 主实验合并。

---

## 6. 缺失比例

正式主实验使用六个缺失比例：

$$
5\%,10\%,15\%,20\%,25\%,30\%
$$

本次 v1.1 更新将 5%、15% 和 25% 纳入主实验，以更细致地观察
missingness-pattern mismatch 随缺失率变化的趋势。所有六档比例均使用相同的
数据划分、masking、插补、候选分类器和模型选择流程。

对于每个序列长度 \(T\)，实际缺失数量始终定义为：

$$
k=\lfloor rT\rfloor
$$

而不是仅依赖近似百分比。

因此实验过程中应记录：

* 原始序列长度 \(T\)；
* 目标缺失比例 \(r\)；
* 实际缺失数量 \(k\)；
* 实际缺失比例 \(k/T\)。

---

## 7. 缺失掩码（Mask）

缺失实验通过 binary mask 实现。

定义：

$$
m_t=
\begin{cases}
1,&\text{observed}\\
0,&\text{missing}
\end{cases}
$$

原始时间序列为：

$$
x=(x_1,\ldots,x_T)
$$

应用 mask 后得到：

$$
x^{mask}
$$

对于缺失位置：

$$
m_t=0
$$

原始值只用于后续评估真实分类标签与实验性能，不得直接用于填补缺失值。

---

## 8. 插补方法

主实验固定使用：

**Linear Interpolation（线性插值）**

对于两个已知观测点之间的缺失值，采用线性插值。

如果：

$$
x_a,\quad x_b
$$

分别位于缺失区间两端，则区间内部根据线性关系估计。

对于序列开头或结尾出现的缺失区间：

* 使用最近的已观测值进行 endpoint extension；
* 不使用被遮蔽的真实值；
* 不利用测试阶段不可获得的信息。

因此整个 preprocessing pipeline 固定为：

$$
\text{masked series}
\rightarrow
\text{linear interpolation}
\rightarrow
\text{classifier}
$$

---

## 9. 插补方法的限制

本研究得到的结论严格针对上述固定插补流程。

研究结果不能自动推广到：

* spline interpolation；
* forward filling；
* mean interpolation；
* Kalman filtering；
* Gaussian Process；
* learned imputation；
* 其他缺失数据恢复模型。

为了控制研究规模，插补方法本身不作为主要研究变量。

---

## 10. 统计分析计划（草案，需在 64 个数据集全部完成前确认）

### 10.1 独立单位

数据集是唯一的独立单位。同一数据集内的 seed、缺失率和缺失模式组合都是重复测量，
必须先在数据集内平均，再跨数据集做区间估计和检验；不得把展开后的行当作独立样本。

### 10.2 两种配对

符号约定：正值表示 mismatch 更差。

* **Target-paired（主分析）**：固定 test pattern，只改变 validation pattern，
  即 PP vs BP、BB vs PB。两者使用同一 test 数据和同一候选集合，因此 test oracle
  相同，有

  $$
  \Delta BA = BA_{matched} - BA_{mismatched} = R_{mismatched} - R_{matched}.
  $$

  两者是同一个效应，不重复报告为两个独立结果。

* **Source-paired（次要分析）**：固定 validation pattern，只改变 test pattern，
  即 PP vs PB、BB vs BP。两者选中的模型相同，差值反映 deployment pattern 对 test
  表现和 oracle 的影响，而不是选择改变。

### 10.3 指标

* 主要指标：target-paired 的 \(\Delta BA\)（selected model 的 test balanced accuracy 差）。
* 次要指标：selection error 差、选择改变率；source-paired 的 regret、selection error、
  selected BA 与 oracle BA 差；四种条件 PP/PB/BP/BB 的数据集层面描述；各分类器被选中
  比例、test-oracle 比例（并列时平分）及被选中时与候选集合 oracle 的差距。

### 10.4 推断

* 每个数据集先得到一个平均差值（n = 64）。
* 报告均值、中位数、按数据集重抽样的 percentile bootstrap 95% 置信区间
  （10,000 次，固定随机种子）、双侧 Wilcoxon signed-rank 检验，以及 mismatch
  更差 / 更好 / 无差异的数据集数量。
* 六档缺失率分别检验，并在每个指标内做 Holm 校正；趋势用每个数据集的
  "差值对缺失率（百分点）"最小二乘斜率，再跨数据集检验斜率。

### 10.5 敏感性分析

* 排除任一侧出现 validation 平票（随机打破）的配对后重复主分析。
* 注明 validation 集较小、部分 seed 下 validation 缺少某一类别的数据集
  （ECG5000、Mallat、WordSynonyms），以及 ItalyPowerDemand 在 5% 时 \(k=1\)，
  此时 point 与 block 不可区分。

### 10.6 补充实验与透明度

* Linear block（第 5 节）使用与主实验相同的 64 个数据集、5 个 seeds、6 个缺失比例、
  3 个候选分类器和分析流程，仅将 circular contiguous block 替换为 non-wrapping
  linear contiguous block。由于 PP 条件不涉及 block missingness，其主实验结果直接复用；
  PB、BP 和 BB 条件重新运行。因此共新增 \(17,280\) 次 classifier evaluations。
  该分析单独报告，不与 circular-block 主实验合并。
* 在 linear-block robustness analysis 中，对于相同的 dataset × seed × missing rate，
  BP 与 BB 共享同一个 linear-block validation mask，PB 与 BB 共享同一个 target
  linear-block test mask；Point missingness 的 mask generation 规则保持不变。
* Linear-block robustness analysis 不修改主实验的 circular-block protocol、主要指标、
  主要统计分析或主实验结果，仅作为独立的 sensitivity analysis。
* 在 64 个数据集完成前曾查看 27 个和 46 个数据集的描述性中期结果；中期结果未做推断检验，
  论文中如实说明。
* 前 9 个数据集由加入预测缓存前的代码生成；已验证两版代码的 test/validation 结果逐条一致。
