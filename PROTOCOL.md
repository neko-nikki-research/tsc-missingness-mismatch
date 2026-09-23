# 缺失模式不匹配对时间序列分类方法选择的影响

## 一项受控实证研究

**English title:**
**The Effect of Missingness-Pattern Mismatch on Method Selection for Time-Series Classification: A Controlled Empirical Study**

**Protocol version:** v1.3
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
`Coffee`, `Computers`, `CricketX`, `CricketY`, `FaceAll`,
`FacesUCR`, `DistalPhalanxOutlineCorrect`,
`DistalPhalanxOutlineAgeGroup`, `DistalPhalanxTW`, `Earthquakes`, `ECG200`,
`ECG5000`, `ECGFiveDays`, `FaceFour`, `FiftyWords`, `Fish`, `GunPoint`,
`Ham`, `Haptics`, `Herring`, `InsectWingbeatSound`, `ItalyPowerDemand`,
`LargeKitchenAppliances`, `Lightning2`, `Lightning7`, `Meat`,
`MedicalImages`, `MiddlePhalanxOutlineCorrect`,
`MiddlePhalanxOutlineAgeGroup`, `MiddlePhalanxTW`, `MoteStrain`, `OliveOil`,
`OSULeaf`, `PhalangesOutlinesCorrect`, `Plane`,
`ProximalPhalanxOutlineCorrect`, `ProximalPhalanxOutlineAgeGroup`,
`ProximalPhalanxTW`, `RefrigerationDevices`, `ScreenType`, `ShapeletSim`,
`ShapesAll`, `SmallKitchenAppliances`, `SonyAIBORobotSurface1`,
`SonyAIBORobotSurface2`, `Strawberry`, `SwedishLeaf`, `Symbols`,
`SyntheticControl`, `ToeSegmentation1`, `ToeSegmentation2`, `Trace`,
`TwoLeadECG`, `Wine`, `WordSynonyms`, `Worms`, `WormsTwoClass`.

该清单在运行任何结果分析前固定。为使 1NN-DTW 在本地可执行，UCR2015 中
21 个样本量与序列长度组合特别大的数据集被基于计算资源预先排除；排除不参考
分类性能、缺失模式或实验结果。

在运行正式结果前的加载预检中，aeon 1.6.0 无法读取 `CricketZ` 的官方测试文件，
并在第 383 条序列报出维度不一致错误；文件的 390 条记录均为长度 300，说明不是
本项目制造缺失或插补造成的问题。为使固定的 64 数据集设计可以完整复现，`CricketZ`
在 v1.3 中被同属 UCR2015、单变量、等长的 `FaceAll` 替换。该替换发生在本批
64 数据集任何正式结果产生之前，不参考任何分类性能。

同样在正式结果产生前的分层划分预检中，`DiatomSizeReduction` 的官方训练集有一个
类别只含 1 条序列，因而无法满足本研究固定的 stratified train/validation split。
v1.3 使用同属 UCR2015、单变量、等长且已通过该预检的 `FacesUCR` 替换它；该决定
不参考任何分类性能、缺失模式或实验结果。

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

这种设计保证每个时间位置在大量重复实验中具有相近的被遮蔽机会。

记该缺失模式为：

$$
B
$$

### 4.1 关于 circular block 的说明

Circular block 用于控制实验中的位置暴露差异。

本研究不会将 circular block 解释为现实世界中某一种特定传感器故障机制。

它主要用于构造一个具有明显局部连续结构、同时避免固定边界效应的受控实验条件。

---

## 5. Supplementary Linear Block

除主实验使用的 circular block 外，再进行普通的线性连续 block masking 作为补充实验。

在线性 block 中：

* 缺失区间必须完全位于 \(1,\ldots,T\) 内；
* 不允许跨越时间序列首尾连接。

该条件用于检查主要观察结果是否依赖于 circular block 的具体构造。

Linear block 不改变主实验问题，也不成为新的主要研究问题。

---

## 6. 缺失比例

正式主实验使用六个缺失比例：

$$
5\%,10\%,15\%,20\%,25\%,30\%
$$

本次 v1.1 更新将 5\%、15\% 和 25\% 纳入主实验，以更细致地观察
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
