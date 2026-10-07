# The Effect of Missingness-Pattern Mismatch on Method Selection for Time-Series Classification

## A Controlled Empirical Study

**Protocol version:** v1.5
**Editorial clarification:** 2026-10-06
**Project status:** Finalized
**Project type:** Controlled empirical study
**Primary task:** Univariate time-series classification
**Primary data source:** UCR2015 archive, accessed through aeon 1.6.0

---

## Protocol Revision History

### v1.1

The main missingness-rate grid was expanded to include 5%, 15%, and 25%, resulting in six prespecified nominal missingness rates: 5%, 10%, 15%, 20%, 25%, and 30%.

### v1.4

The primary experiment was finalized using 64 fixed UCR2015 univariate, equal-length datasets and the circular point/block \(2\times2\) validation-test design.

### v1.5

A supplementary robustness analysis was added to replace circular contiguous block masking with non-wrapping linear contiguous block masking. The primary circular-block analysis, its estimand, and its statistical analysis remain unchanged. The v1.5 robustness analysis is reported separately and is not pooled with the primary analysis.

### Editorial clarification: 2026-10-06

This revision clarifies the documented v1.4 and v1.5 execution paths, classifier preprocessing and internal fitting rules, dataset exclusions, aggregation, and interpretation boundaries. It corrects the circular-block indexing example. It does not change the dataset set, masking distributions, candidate set, primary estimand, or frozen results. The numerical-stability follow-up in Section 15.5 is identified as an audit recommendation rather than an implemented change to the original analysis.

### Versioned reanalysis: 2026-10-07

The numerical-stability follow-up recommended in Section 15.5 was implemented as a versioned reanalysis (`analysis_v2/`) of the unchanged frozen result files; the original `analysis/` outputs are retained. Trend slopes are computed in closed form, and dataset-level values and slopes are rounded to 12 decimal places before they are summarized, so that values that are equal or zero in exact arithmetic are not separated by floating-point error. Means, intervals, and direction counts are unchanged beyond 10^-12. The affected Wilcoxon results, including two reported Holm-adjusted rate-specific \(p\)-values that changed in the second significant digit, are listed in the result-set READMEs; no conclusion changed. Post hoc sensitivity checks reported with the results are reproduced by `scripts/post_hoc_checks.py` and are not part of the prespecified analysis.

---

# 1. Research Question

This study investigates a specific model-selection problem in time-series classification:

> At a fixed missingness rate, if classifier selection is performed on validation data with one missingness pattern while the target test data exhibit a different missingness pattern, does the mismatch affect the selected classifier and its test performance?

The study focuses specifically on a mismatch between the missingness pattern observed during validation and the missingness pattern encountered at deployment.

The primary comparison is between matched and mismatched validation conditions for the same target test condition. In the primary experiment, the two missingness patterns are random point missingness and circular contiguous block missingness.

The four primary validation-test conditions are:

| Validation pattern | Test pattern   | Condition |
| ------------------ | -------------- | --------- |
| Point              | Point          | \(PP\)    |
| Point              | Circular block | \(PB\)    |
| Circular block     | Point          | \(BP\)    |
| Circular block     | Circular block | \(BB\)    |

For a fixed target test condition, the matched and mismatched conditions differ only in the validation missingness pattern used for classifier selection.

The primary research question therefore separates two quantities that are often conflated:

1. the predictive performance of individual classifiers under different missingness conditions; and
2. the model-selection decision produced by a validation procedure when the validation condition differs from the target deployment condition.

The primary estimand is the paired difference in test balanced accuracy between classifiers selected under matched and mismatched validation conditions.

---

# 2. Study Scope and Data

The study considers:

* supervised univariate time-series classification;
* equal-length time series;
* originally complete time-series datasets;
* publicly available benchmark data;
* datasets from the UCR2015 archive.

The study does not address multivariate time series, irregularly sampled series, MAR or MNAR missingness mechanisms, domain-specific missingness mechanisms, foundation models, Transformer-based large models, new classifier architectures, new missingness metrics, or new imputation algorithms.

## 2.1 Main Experimental Dataset Set

The primary experiment uses the following 64 fixed univariate, equal-length datasets from the UCR2015 archive:

`Adiac`, `ArrowHead`, `Beef`, `BeetleFly`, `BirdChicken`, `Car`, `CBF`, `Coffee`, `Computers`, `CricketX`, `CricketY`, `CricketZ`, `FaceAll`, `DistalPhalanxOutlineCorrect`, `DistalPhalanxOutlineAgeGroup`, `DistalPhalanxTW`, `Earthquakes`, `ECG200`, `ECG5000`, `ECGFiveDays`, `FaceFour`, `FacesUCR`, `Fish`, `GunPoint`, `Ham`, `Haptics`, `CinCECGTorso`, `InsectWingbeatSound`, `ItalyPowerDemand`, `LargeKitchenAppliances`, `Lightning2`, `Lightning7`, `Meat`, `MedicalImages`, `MiddlePhalanxOutlineCorrect`, `MiddlePhalanxOutlineAgeGroup`, `MiddlePhalanxTW`, `MoteStrain`, `OliveOil`, `OSULeaf`, `PhalangesOutlinesCorrect`, `Plane`, `ProximalPhalanxOutlineCorrect`, `ProximalPhalanxOutlineAgeGroup`, `Mallat`, `RefrigerationDevices`, `ScreenType`, `ShapeletSim`, `ShapesAll`, `SmallKitchenAppliances`, `SonyAIBORobotSurface1`, `SonyAIBORobotSurface2`, `Strawberry`, `SwedishLeaf`, `Symbols`, `SyntheticControl`, `ToeSegmentation1`, `ToeSegmentation2`, `Trace`, `TwoLeadECG`, `Wine`, `WordSynonyms`, `Worms`, `WormsTwoClass`.

This dataset list is fixed before formal result analysis.

An initial resource-based screening omitted 21 UCR2015 datasets whose combinations of sample size and sequence length made 1-nearest-neighbour DTW computationally impractical under the available local resources. Four of these datasets were subsequently included as replacements for four unusable datasets in the initial selection, as described in Section 2.2. The final exclusions therefore comprise 17 datasets omitted for computational reasons and four that could not be loaded or split under the specified procedure. Neither screening nor replacement used classifier performance or missingness outcomes.

## 2.2 Dataset Preflight and Replacements

Before the formal experiment, all configured datasets underwent loading and splitting preflight.

Seven local aeon cache files were found to be truncated at the final series, producing dimensional inconsistencies or incorrect class counts. The affected dataset directories were removed and re-downloaded through aeon. `CricketZ`, `MiddlePhalanxOutlineAgeGroup`, and `MiddlePhalanxTW` subsequently passed the full loading and stratified-splitting preflight.

After re-downloading, `DiatomSizeReduction` and `FiftyWords` each contained a class with only one official training instance and therefore could not support the prespecified stratified train-validation split. `Herring` and `ProximalPhalanxTW` remained unreadable by aeon 1.6.0.

Before the formal v1.4 results were generated, these four datasets were replaced, respectively, by `FaceAll`, `FacesUCR`, `CinCECGTorso`, and `Mallat`. The replacement datasets satisfy the same structural eligibility requirements: UCR2015 membership, univariate format, equal length, successful loading, and successful stratified splitting. No classifier performance, missingness result, or other experimental outcome was used to determine these replacements.

For each series,

$$
x=(x_1,x_2,\ldots,x_T),
$$

where \(T\) is the number of time points and \(x_t\) is the observation at time point \(t\). Each series has an associated class label.

---

# 3. Experimental Data Splitting

For each dataset and random seed, the official UCR training set is divided into:

1. a complete training subset; and
2. a held-out validation subset.

The target validation fraction is 25%; the realized fraction can differ slightly because instance counts are integers.

The split is stratified by class label and uses random seeds

$$
\{1,2,3,4,5\}.
$$

Formally,

$$
D_{\mathrm{UCR,train}}
=
D_{\mathrm{train}}
\cup
D_{\mathrm{val}},
$$

with approximately 75% of the official training instances assigned to \(D_{\mathrm{train}}\) and 25% assigned to \(D_{\mathrm{val}}\).

The official UCR test set is retained as the deployment test set:

$$
D_{\mathrm{test}}=D_{\mathrm{UCR,test}}.
$$

Training data remain complete and unchanged throughout the experiment. Synthetic missingness is introduced only into the held-out validation subset and the official test set.

The same train-validation split is reused across all missingness rates and missingness-pattern conditions for a given dataset and seed.

This design isolates validation-to-deployment missingness mismatch from changes in the training distribution.

---

# 4. Missingness Generation

Missingness is imposed through a binary mask.

For a time series of length \(T\) and nominal missingness rate \(r\), the number of masked observations is

$$
k=\lfloor rT\rfloor.
$$

The same \(k\) is used for all missingness patterns for a given series length and nominal missingness rate.

Define the mask

$$
m_t=
\begin{cases}
1,&\text{observed},\\
0,&\text{missing}.
\end{cases}
$$

The masked series \(x^{mask}\) is obtained by replacing observations at positions with \(m_t=0\) by missing values.

The original values at masked positions are not supplied to the imputation procedure. They are retained only in the underlying evaluation framework as the values that generated the masked input; they are never used as preprocessing information.

For every dataset, seed, and missingness rate, validation and test masks are generated separately using deterministic seed offsets: experimental seed + 1 for validation and experimental seed + 2 for test masking. For a fixed series-array shape, pattern, rate, and seed, mask generation is reproducible. Conditions with an unchanged validation or test pattern share the corresponding masked data. Mask placement does not depend on series values or class labels.

All candidate classifiers within the same experimental cell receive the same validation and test masks.

---

# 5. Missingness Patterns

The primary experiment uses two missingness patterns:

* random point missingness, denoted \(P\);
* circular contiguous block missingness, denoted \(B\).

Both remove exactly \(k\) observations from each series at the same nominal missingness rate.

## 5.1 Random Point Missingness

Point missingness represents randomly scattered missing observations.

For a series of length \(T\), \(k\) distinct time positions are sampled uniformly without replacement from the \(T\) available positions.

Thus,

$$
|M_P|=k,
$$

and every selected position is masked exactly once.

Point missingness is intended to create isolated or scattered missing observations rather than a contiguous temporal gap.

We denote this pattern by

$$
P.
$$

## 5.2 Circular Contiguous Block Missingness

The primary block condition uses a circular contiguous block.

A starting position \(s\) is sampled uniformly from the \(T\) time points. Starting from \(s\), \(k\) consecutive positions are masked under circular indexing.

Using zero-based indexing, the masked set is

$$
M_B
=
\{(s+j)\bmod T:j=0,\ldots,k-1\}.
$$

Consequently, a block may cross the sequence boundary. For example, when \(T=20\) and \(k=5\), a valid block may be

$$
(17,18,19,0,1).
$$

Circular placement gives every time point the same marginal probability of being included in a randomly positioned block of length \(k\).

Circular indexing is used only for mask construction. It does not imply that the underlying time series are periodic, and no periodicity assumption is made during imputation or classification.

The circular block condition is intended as a controlled construction of temporally contiguous missingness that avoids systematic preference for interior or boundary positions. It is not interpreted as a specific real-world sensor-failure mechanism.

We denote this pattern by

$$
B.
$$

---

# 6. Supplementary Linear-Block Robustness Analysis

To determine whether the main findings depend on the specific construction of circular block missingness, a separate robustness analysis uses non-wrapping linear contiguous blocks.

In the linear-block condition, the masked interval must lie completely within the original sequence. Using zero-based indexing:

$$
M_L=\{s,s+1,\ldots,s+k-1\},
$$

where

$$
s\sim\mathrm{Uniform}\{0,\ldots,T-k\}.
$$

A linear block therefore cannot cross the beginning or end of the sequence.

This change also alters the marginal masking probability of positions near the sequence boundaries relative to circular masking. That difference is part of the block-construction robustness comparison and is not treated as a separate experimental factor.

The circular-block experiment remains the prespecified primary analysis. Linear-block results are supplementary and are analyzed separately.

## 6.1 Supplementary Experimental Design

The linear-block robustness analysis keeps the following unchanged:

* the same 64 datasets;
* the same five random seeds;
* the same six nominal missingness rates;
* the same train-validation-test splitting procedure;
* the same three candidate classifiers;
* the same classifier parameters;
* the same linear interpolation procedure;
* the same validation-based model-selection rule;
* the same evaluation metrics;
* the same dataset-level statistical analysis.

Only the block-masking construction is changed from circular contiguous blocks to non-wrapping linear contiguous blocks.

The supplementary run evaluates:

* Point validation \(\rightarrow\) Linear-block test (\(PL\));
* Linear-block validation \(\rightarrow\) Point test (\(LP\));
* Linear-block validation \(\rightarrow\) Linear-block test (\(LL\)).

The Point-to-Point condition does not involve block masking and is therefore reused from the primary experiment rather than rerun.

The newly generated supplementary conditions contain

$$
64\times5\times6\times3\times3
=
17,280
$$

classifier-condition result records.

For each dataset, seed, and missingness rate:

* the \(LP\) and \(LL\) conditions share the same linear-block validation mask;
* the \(PL\) and \(LL\) conditions share the same linear-block test mask;
* the \(PP\) and \(PL\) conditions use the same point-validation mask generation rule;
* the \(PP\) and \(LP\) conditions use the same point-test mask generation rule.

The supplementary result set is assembled only after point-side validation and test balanced accuracies generated by the new run have been checked against the corresponding primary values for every candidate, dataset, seed, and rate, using an assembly tolerance of 10^-12. This check compares recorded metrics; it does not directly compare mask arrays, prediction arrays, or fitted estimator objects. The assembled analysis contains the same four-condition structure as the primary analysis but is based on linear blocks for all conditions that involve block missingness.

The robustness analysis is never pooled with the circular-block primary experiment.

---

# 7. Missingness Rates

The primary experiment uses six nominal missingness rates:

$$
r\in
\{0.05,0.10,0.15,0.20,0.25,0.30\}.
$$

For every series,

$$
k=\lfloor rT\rfloor.
$$

Thus, nominal percentage and actual number of missing observations are not treated as interchangeable. Both the nominal rate and the realized rate

$$
\frac{k}{T}
$$

are recorded.

For each experimental series, the protocol records:

* the original sequence length \(T\);
* the nominal missingness rate \(r\);
* the number of missing observations \(k\);
* the realized missingness rate \(k/T\).

A boundary case occurs when \(k=1\). In this case, point and contiguous-block masking are distributionally equivalent because both select one time point. This case is treated as a non-discriminating boundary condition rather than evidence that the two missingness constructions differ.

In particular, `ItalyPowerDemand` has \(k=1\) at the 5% nominal rate. The point and block mechanisms therefore have the same distribution in that condition, although their separately generated realized masks need not be identical.

---

# 8. Imputation

All masked validation and test series are processed using a single fixed imputation method:

**linear interpolation**.

For an internal missing interval bounded by observed values \(x_a\) and \(x_b\), missing observations are estimated according to the linear relationship between the two observed endpoints.

For missing values at the beginning or end of a series, the nearest observed value is extended to the corresponding endpoint. This is equivalent to the endpoint behavior of the interpolation implementation used in the experiment.

The imputation procedure operates independently on each series and uses only the observed values of that series.

It does not use:

* masked original values;
* class labels;
* other validation instances;
* other test instances;
* statistics estimated from the official test set.

The project applies masking and interpolation to the arrays returned by the aeon loader without adding series normalization before or after masking. Classifier-specific feature scaling is part of the MiniROCKET pipeline described in Section 9.2.

The preprocessing pipeline is therefore

$$
\text{masked series}
\rightarrow
\text{linear interpolation}
\rightarrow
\text{classifier}.
$$

The imputation method is fixed and is not an experimental factor in the primary study.

The findings therefore do not automatically generalize to other imputation procedures such as spline interpolation, forward filling, mean imputation, Kalman filtering, Gaussian processes, learned imputers, or other missing-data reconstruction methods.

---

# 9. Classification Methods

The candidate set contains three prespecified classification pipelines. The purpose of this candidate set is to provide methods representing different TSC approaches rather than to establish a comprehensive ranking of all available TSC algorithms.

## 9.1 1-NN-DTW

The first candidate is one-nearest-neighbour classification using dynamic time warping as the distance measure.

The number of neighbours is fixed at

$$
k_{\mathrm{NN}}=1.
$$

The implementation uses aeon's DTW defaults with no warping-window or Itakura constraint. No hyperparameter tuning is performed on the held-out validation subset for this classifier.

## 9.2 MiniROCKET-Ridge

The second candidate uses MiniROCKET followed by a Ridge classifier.

The configured MiniROCKET kernel-count parameter is fixed at

$$
10,000.
$$

The linear classifier is `RidgeClassifierCV` with ten prespecified regularization values:

$$
\alpha\in
\{10^{-3},10^{-3+6/9},\ldots,10^3\}.
$$

The aeon 1.6.0 classifier applies `StandardScaler(with_mean=False)` to the transformed features before the linear classifier. Ridge regularization is selected within the complete fitting subset using `RidgeClassifierCV` with `cv=None` and `scoring=None`: the default efficient leave-one-out procedure and its negative-mean-squared-error criterion. The held-out validation labels and official test labels are not used for this internal fitting operation. Candidate definitions, the regularization grid, and internal fitting rules are fixed; the fitted regularization value can vary across training splits.

The external validation set remains reserved for choosing among the three candidate pipelines.

## 9.3 Statistical-Feature Random Forest

The third candidate first transforms each series into ten fixed statistical features:

1. mean;
2. standard deviation;
3. minimum;
4. maximum;
5. median;
6. interquartile range;
7. linear slope;
8. first value;
9. last value;
10. lag-1 autocorrelation.

These features are then classified using a Random Forest with

$$
500
$$

trees.

The Random Forest random state is determined by the experimental seed.

No validation-based hyperparameter tuning is performed for this pipeline.

---

# 10. Validation-Based Model Selection

For each dataset, seed, missingness rate, validation pattern, and target test pattern, each candidate classifier is fitted using only the complete training subset.

Because the fitting subset remains unchanged across missingness rates and validation/test patterns, a fitted candidate can be reused across conditions. The original primary runner refitted candidates for each condition and introduced prediction caching after the first nine datasets. The supplementary runner and current implementation fit each candidate once per dataset and seed and reuse it across rates and patterns. These execution paths use the same fitting inputs, candidate specifications, and selection rule; their verification is described in Section 19.

For each validation condition, predictions are produced on the masked and imputed validation subset.

Classifier selection is based exclusively on validation balanced accuracy:

$$
\hat{j}
=
\arg\max_{j\in\mathcal{C}}
BA^{val}_j,
$$

where \(\mathcal{C}\) is the three-classifier candidate set.

The official test labels and test balanced accuracies do not participate in classifier selection.

## 10.1 Validation Ties

A validation tie is defined when multiple candidates have validation balanced accuracy values within a numerical tolerance of

$$
10^{-12}
$$

of the maximum validation balanced accuracy.

When a validation tie occurs, one of the tied candidates is selected uniformly at random using a deterministic pseudorandom generator whose seed is derived solely from validation-side experiment metadata:

* dataset;
* random seed;
* missingness rate;
* validation pattern;
* imputer.

The classifier names are sorted before the tied choice is made.

This tie-breaking rule does not use any test outcome.

The number of validation-tied candidates is retained in the result files.

---

# 11. Experimental Procedure

For each dataset and random seed:

1. Load the official UCR training and test sets.
2. Validate that both sets are univariate, equal-length, and originally complete.
3. Split the official training set into complete training and held-out validation subsets using the fixed stratified 25% validation split.
4. Fit the three candidate classifiers on the complete training subset using the execution path described in Sections 10 and 19.
5. For each nominal missingness rate, generate the required validation and test masks.
6. Impute the masked validation and test series independently using linear interpolation.
7. Obtain validation predictions for each candidate classifier.
8. Select the classifier using validation balanced accuracy only.
9. Obtain test predictions for each candidate classifier under the target test pattern.
10. Evaluate the selected classifier on the official test set.
11. Compute the retrospective candidate-set test oracle.
12. Record classification performance, selection, selection error, and regret.

Validation predictions do not depend on the target test pattern, and test predictions do not depend on the validation pattern. Where prediction caching is used, the corresponding predictions are reused across target or source conditions, as described in Sections 10 and 19.

This prediction reuse is an implementation optimization that does not change the experimental design.

---

# 12. Primary \(2\times2\) Experimental Design

Let the first pattern label denote the validation pattern and the second pattern label denote the test pattern.

The primary conditions are:

$$
PP,\quad PB,\quad BP,\quad BB.
$$

The primary analysis is **target-paired**.

For a fixed target test pattern, the matched and mismatched validation conditions are paired on the same dataset, random seed, missingness rate, imputer, and test masks.

### Point test target

For point-masked deployment:

$$
PP
\quad\text{vs.}\quad
BP.
$$

Here, validation changes from Point to Block while the target test condition remains Point.

### Circular-block test target

For block-masked deployment:

$$
BB
\quad\text{vs.}\quad
PB.
$$

Here, validation changes from Block to Point while the target test condition remains Circular Block.

In each target-paired comparison, the following are held fixed:

* dataset;
* train-validation split;
* random seed;
* nominal missingness rate;
* target test instances;
* target test masks;
* imputation method;
* candidate classifier set;
* fitted classifier parameters.

Only the validation missingness pattern differs.

---

# 13. Secondary Source-Paired Analysis

A secondary analysis fixes the validation pattern while changing the target test pattern.

The two source-paired comparisons are:

$$
PP\quad\text{vs.}\quad PB
$$

and

$$
BB\quad\text{vs.}\quad BP.
$$

Because the validation data do not depend on the target test pattern, the selected classifier is the same within each source-paired comparison under the deterministic selection rule.

The source-paired analysis therefore characterizes the effect of deployment missingness pattern on:

* selected-model test balanced accuracy;
* candidate-set oracle performance;
* selection regret;
* selection error.

This analysis is secondary and does not serve as the primary estimate of validation-pattern mismatch.

---

# 14. Evaluation Metrics

## 14.1 Balanced Accuracy

Balanced accuracy is the primary classification-performance metric.

For the \(C\) classes represented in the evaluated ground-truth labels,

$$
BA
=
\frac{1}{C}
\sum_{c=1}^{C}
\frac{TP_c}{TP_c+FN_c}.
$$

Balanced accuracy is used for:

* classifier selection;
* candidate-set oracle identification;
* selection regret;
* the primary test-performance comparison.

Macro-F1 is recorded as an additional descriptive classification metric but is not used for classifier selection, oracle identification, or the primary statistical analysis.

## 14.2 Primary Mismatch Effect

The sign convention is:

> **positive values indicate worse performance under mismatched validation.**

For point-masked test data,

$$
\Delta BA_P
=
BA_{PP}-BA_{BP}.
$$

For circular-block test data,

$$
\Delta BA_B
=
BA_{BB}-BA_{PB}.
$$

The primary overall target-paired effect is obtained from dataset-level averages across the prespecified target-pattern strata and missingness rates.

## 14.3 Selection Regret

For a given experimental condition, the retrospective candidate-set test oracle is

$$
BA^{oracle}
=
\max_{j\in\mathcal{C}}
BA^{test}_j.
$$

Selection regret is

$$
R
=
BA^{oracle}
-
BA^{selected}.
$$

The oracle is computed only after test evaluation and is used only as a retrospective evaluation reference.

It is not involved in classifier selection.

When multiple candidates tie for the maximum test balanced accuracy, all tied classifier names are retained as oracle members. An oracle tie is not itself counted as a selection error.

## 14.4 Selection Error

Selection error is defined as:

$$
I
\left(
BA^{selected}
<
BA^{oracle}-10^{-12}
\right).
$$

Thus a selected classifier is considered erroneous only when its test balanced accuracy is strictly below the candidate-set oracle beyond the numerical tolerance.

A classifier that ties the candidate-set oracle is not counted as a selection error.

## 14.5 Selection Change

For a matched/mismatched target-paired comparison, we also record whether the selected classifier identity changes.

This indicator is descriptive of the model-selection decision and is distinct from test-performance loss.

A change in classifier identity does not necessarily imply a difference in test balanced accuracy if the candidates perform equally on the same target test set.

---

# 15. Statistical Analysis

## 15.1 Independent Unit

The dataset is the analysis unit for cross-task inference. Independence and exchangeability are analysis assumptions: some included tasks belong to related families, and the analysis does not explicitly model that dependence.

Seeds, missingness rates, and missingness-pattern conditions are repeated measurements within a dataset. They are therefore aggregated within dataset before across-dataset inference.

The analysis does not treat the individual seed-condition rows as independent observations.

The inferential sample size for the primary dataset-level analysis is therefore

$$
n=64.
$$

## 15.2 Dataset-Level Aggregation

For each dataset, repeated observations are averaged over the relevant combinations of:

* random seed;
* missingness rate;
* target-pattern stratum where applicable.

This produces one dataset-level value for each reported analysis.

For the primary overall target-paired analysis, the paired effects are averaged within dataset across the two target-pattern strata and six missingness rates before the cross-dataset analysis.

## 15.3 Primary Inference

For each primary outcome, the following are reported:

* mean dataset-level effect;
* median dataset-level effect;
* 95% percentile bootstrap confidence interval;
* number of datasets where mismatch is worse;
* number of datasets where mismatch is better;
* number of datasets with no difference;
* two-sided Wilcoxon signed-rank test.

Bootstrap confidence intervals are generated by resampling datasets with replacement. They describe variation under resampling of the included tasks, treating datasets as exchangeable units, rather than probability-sampling uncertainty for all TSC problems. Test instances and mask realizations are not separately resampled.

The number of bootstrap resamples is

$$
10,000.
$$

The bootstrap random seed is fixed at

$$
20260928.
$$

The Wilcoxon signed-rank test is applied to dataset-level values rather than individual repeated measurements. The implemented call uses `alternative="two-sided"` and `zero_method="wilcox"`; `method` and `correction` are not explicitly overridden and therefore use the installed SciPy defaults. Exactly zero differences are omitted by the test. In the versioned reanalysis of 2026-10-07, dataset-level values are rounded to 12 decimal places before the test, so that values that are zero in exact arithmetic are omitted as zeros rather than ranked as floating-point residues. Direction counts use a separate tolerance of 10^-9, and that tolerance does not otherwise recode the test inputs. When all dataset-level values lie within that tolerance of zero, the implementation reports p = 1. The signed-rank test is not specifically a test of the arithmetic mean; a location-shift interpretation requires the usual assumptions about the difference distribution.

No directional alternative is assumed for the primary hypothesis test.

## 15.4 Missingness-Rate Analysis

The primary mismatch effect is also summarized separately at each of the six nominal missingness rates.

The six rate-specific Wilcoxon \(p\)-values are adjusted within each metric using the Holm procedure.

Rate-specific tests are therefore corrected across the six missingness rates for a given signed metric. This is not a joint correction across the overall, stratum-specific, trend, and supplementary analyses. Selection-change frequencies are descriptive; generic signed-rank output for that nonnegative quantity is not interpreted as evidence of performance harm.

## 15.5 Missingness-Rate Trend

To quantify how the mismatch effect changes with missingness burden, the analysis first computes one target-paired effect per dataset at each nominal missingness rate.

For each dataset, a least-squares slope is then fitted:

$$
\Delta(r)
=
\beta_0+\beta_1(100r),
$$

where \(\beta_1\) represents the change in the mismatch effect per one percentage-point increase in nominal missingness.

The six rate-specific observations are averaged within dataset across target-pattern strata before the slope is estimated.

The resulting dataset-level slopes are then summarized across the 64 datasets using the same bootstrap and Wilcoxon framework.

Editorial audit note (2026-10-06): reanalysis reproduced the reported balanced-accuracy trend tests, but small changes in secondary selection-error and selection-change trend p-values were observed across numerical environments. Floating-point least-squares estimates can separate theoretical ties or turn a theoretical zero into a very small nonzero slope. A numerical-stability improvement is recommended for any versioned reanalysis, with its treatment of theoretical ties and zero slopes documented explicitly. Such a change has not been applied to the frozen results in this editorial revision.

Implementation note (2026-10-07): the recommended change was implemented in the versioned reanalysis described in the revision history. Slopes are computed in closed form and rounded to 12 decimal places; the frozen result files and the original analysis outputs are unchanged.

## 15.6 Sensitivity to Validation Ties

A primary sensitivity analysis excludes any target-paired observation for which either side contains a validation tie.

The overall target-paired analysis is then repeated using only pairs without validation ties.

The primary estimand and dataset-level analysis unit remain unchanged. The tie-excluded summary is a sensitivity analysis restricted to the retained tie-free pairs, with dataset-level means recomputed from those pairs and datasets without retained pairs omitted. It therefore does not reproduce the original observation set or weighting.

Several datasets have relatively small classes in their official training sets, and for some random seeds the held-out validation subset does not contain every class. These include `ECG5000`, `Mallat`, and `WordSynonyms`. These cases are retained because they arise from the prespecified stratified split and are handled by the same balanced-accuracy computation for all experimental conditions.

At the 5% nominal rate, `ItalyPowerDemand` has \(k=1\). Point and block masking are distributionally identical in this boundary case.

---

# 16. Descriptive Classifier Analysis

For each validation-test condition, the study reports the proportion of conditions in which each candidate classifier is selected.

The study also reports candidate-set oracle membership.

When \(q\) classifiers tie for the test oracle, each tied classifier receives an oracle share of

$$
\frac{1}{q}.
$$

Classifier selection and oracle shares are summarized with equal weighting at the dataset level rather than by treating all repeated seed-condition rows as independent datasets.

For each classifier, mean regret when selected is computed by first averaging its regret over the corresponding selected conditions within each dataset, then averaging equally over the datasets in which it was selected at least once.

Analyses explicitly pooled over changed-selection target pairs are descriptive conditional summaries rather than dataset-level inferential tests. Each target-paired selection-change indicator is identical across the two target strata because selection depends only on validation.

These analyses are descriptive and do not redefine the primary estimand.

---

# 17. Supplementary Linear-Block Statistical Analysis

The linear-block robustness analysis uses the same dataset-level statistical framework as the primary circular-block analysis.

For point-test deployment, the supplementary target-paired effect is

$$
\Delta^{linear}_{P}
=
BA_{PP}-BA_{LP},
$$

where \(L\) denotes non-wrapping linear-block validation.

For linear-block-test deployment, the supplementary target-paired effect is

$$
\Delta^{linear}_{L}
=
BA_{LL}-BA_{PL}.
$$

The supplementary analysis reports:

* selected-model test balanced accuracy difference;
* selection-error difference;
* selection-change rate;
* results by test pattern;
* results by missingness rate;
* validation-tie sensitivity.

The primary circular-block results are retained as the main estimates. Linear-block results are presented as a separate supplementary robustness analysis; the datasets and primary PP condition are shared with the primary experiment.

The two block constructions are not pooled because they correspond to different mask-generating distributions. The study does not perform a formal test of the difference between their effects. A significant effect under one construction and a nonsignificant effect under the other do not establish a statistically significant difference between constructions.

The purpose of the linear-block analysis is not to establish linear blocks as a new primary experimental factor. It is to assess whether conclusions obtained under the circular contiguous-block construction persist when the construction is changed to a non-wrapping contiguous block.

---

# 18. Experimental Integrity and Reproducibility

The following invariants are required throughout the experiment.

### Training isolation

The official UCR training data used for fitting remain complete and unmasked.

### Validation-only selection

Only validation balanced accuracy determines classifier selection.

### Test isolation

Official test labels are never used for model selection or tuning. They are used only for candidate evaluation and retrospective candidate-set oracle calculation. Their computational availability does not make them inputs to the validation-only selection rule.

### Shared masks

All candidate classifiers within an experimental condition receive identical validation and test masks.

### Separate validation and test masking

Validation and test masks are generated separately using distinct seed offsets.

### Pattern invariance

Validation predictions depend on the validation pattern but not on the target test pattern. Test predictions depend on the target test pattern but not on the validation pattern.

### Candidate-set invariance

The same three classifier pipelines are used in every formal experimental condition.

### Imputation isolation

Imputation is performed independently within each series and uses only that series' observed values.

### Raw-result reconstruction

Selection, oracle membership, selection error, and regret must be reproducible from the raw classifier-level results.

### Dataset completeness

Formal confirmatory analysis requires all 64 configured datasets to be complete. Partial runs may be inspected descriptively, but they are not treated as final inferential results.

---

# 19. Computational Implementation

The formal experiments are implemented in Python using aeon 1.6.0, NumPy, pandas, scikit-learn, SciPy, Matplotlib, PyYAML, and pytest.

The primary main-study configuration fixes:

* five seeds: 1, 2, 3, 4, 5;
* six missingness rates: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30;
* validation size: 0.25;
* imputer: linear;
* classifiers: DTW, MiniROCKET, and statistical-feature Random Forest;
* Random Forest: 500 trees;
* MiniROCKET: 10,000 kernels;
* main-study computational parallelism: 18 jobs.

The supplementary linear-block run uses the same scientific parameters and 24 jobs for computational parallelism. The parallelism setting is an implementation detail and is not treated as an experimental factor.

The primary frozen results were generated by the original runner, which fitted candidates separately for each condition. Prediction caching was introduced after the first nine primary datasets. The supplementary runner and current implementation fit each candidate once per dataset and seed, reuse the fitted estimator across rates and patterns, and batch 1NN-DTW predictions through aeon's neighbour-search routine. The wrapper retains the distance calculation and nearest-neighbour tie rule.

The repository includes a regression test comparing batched 1NN-DTW predictions with the aeon reference, including tied distances. Archive-level verification covers the validation- and test-score invariances in the first nine primary datasets and agreement of recorded point-side balanced accuracies across all candidates, datasets, seeds, and rates. These checks support consistency of the execution paths; metric agreement is not a direct comparison of every prediction array or fitted estimator object.

The optimized implementation reuses validation predictions across target patterns and test predictions across source patterns within each missingness rate.

The benchmark implementation stores both classifier-level raw results and condition-level selection results. Timing fields for reused fits and predictions are repeated per condition and must not be summed as additive wall-clock costs.

The supplied requirements file pins aeon to 1.6.0 and specifies supported ranges for other dependencies. The supplied run manifests do not contain complete installed-package version snapshots. Exact historical versions should be recovered and documented where available; supported ranges alone do not uniquely reconstruct the original environment.

---

# 20. Study Outputs and Verification

For the primary experiment,

$$
64\times5\times6\times4\times3
=
23,040
$$

classifier-condition result records are expected.

The corresponding selection-level table contains

$$
64\times5\times6\times4
=
7,680
$$

conditions.

For the supplementary linear-block run, the newly executed conditions contain

$$
64\times5\times6\times3\times3
=
17,280
$$

classifier-condition result records.

The assembled linear-block analysis reuses the primary \(PP\) condition and therefore has the same four-condition record structure as the primary experiment.

Before formal analysis, result files must satisfy:

* expected row counts;
* unique experiment keys;
* no missing metric values;
* exactly the configured classifier set for every experimental condition;
* valid balanced-accuracy values;
* consistency of the configured missingness counts;
* successful recomputation of selection, oracle, selection error, and regret from raw classifier results.

For paired analyses, the shared-target oracle invariant must hold:

$$
BA^{oracle}_{matched}
=
BA^{oracle}_{mismatched}.
$$

Consequently, for target-paired comparisons,

$$
\Delta BA
=
BA_{matched}-BA_{mismatched}
$$

must equal

$$
R_{mismatched}-R_{matched}.
$$

Any violation beyond the prespecified numerical tolerance invalidates the paired result and must be investigated before analysis continues.

---

# 21. Interim Results and Study Conduct

During the primary 64-dataset run, two frozen partial checkpoints containing 27 and 46 completed datasets were inspected descriptively.

These checkpoints are ordered prefixes of the fixed dataset list rather than random or representative samples.

No inferential analysis is based on these incomplete checkpoints. Formal confirmatory analysis is performed on the completed 64-dataset experiment.

The v1.5 linear-block robustness analysis is supplementary to the completed v1.4 primary experiment. It does not replace, modify, or redefine the primary circular-block analysis.

---

# 22. Interpretation Boundaries

The study is designed to answer a controlled question under a fixed experimental pipeline. Its conclusions are therefore limited to the conditions actually studied.

In particular, the results should not automatically be generalized to:

* multivariate time-series classification;
* irregular or unequal-length time series;
* real-world missingness mechanisms such as MAR or MNAR;
* informative or value-dependent missingness;
* alternative missingness patterns outside the tested point and contiguous-block constructions;
* alternative imputation methods;
* classifier candidates outside the three prespecified pipelines;
* deployment settings in which the training distribution itself is affected by missingness.

The candidate-set test oracle is the highest observed score among the three prespecified classifiers on a finite target test set. It is a retrospective evaluation reference, not a population-optimal method or a global oracle over all possible TSC algorithms. Selection error denotes observed candidate-set suboptimality beyond numerical tolerance, not a statistically established population-performance difference.

An average mismatch effect close to zero does not imply that individual changed selections perform similarly. Positive and negative paired differences can offset each other. A source-paired regret increase is relative to a target-specific oracle and does not by itself imply a decrease in absolute test balanced accuracy.

Associations involving validation-set size or candidate dominance are descriptive or mechanistic hypotheses unless tested directly by a corresponding experimental comparison.

The circular-block construction is a controlled masking mechanism rather than a claim about a particular real-world missingness process.

The linear-block experiment assesses robustness to a different contiguous-block construction rather than proving invariance to all possible block-missingness mechanisms.

---

# 23. Primary Analysis Summary

The primary analysis tests whether, at a fixed nominal missingness rate and fixed target test condition, changing only the validation missingness pattern changes the classifier-selection decision and the resulting test performance.

The primary effect is:

$$
\Delta BA
=
BA_{matched}
-
BA_{mismatched}.
$$

The primary analysis unit is the dataset, with cross-dataset independence and exchangeability treated as analysis assumptions as described in Section 15.1.

The primary experiment uses:

* 64 fixed UCR2015 univariate, equal-length datasets;
* five random train-validation splits;
* six nominal missingness rates from 5% to 30%;
* two missingness patterns;
* four validation-test conditions;
* three prespecified classifier pipelines;
* fixed linear interpolation;
* validation balanced accuracy for method selection;
* candidate-set test oracle and selection regret for post-hoc evaluation;
* dataset-level bootstrap confidence intervals and Wilcoxon signed-rank tests.

The supplementary v1.5 analysis replaces circular contiguous block masking with non-wrapping linear contiguous block masking while preserving the same scientific and statistical framework.

The central object of inference is therefore not the standalone effect of missing values on classifier accuracy, but the effect of **validation-deployment missingness-pattern mismatch on validation-based method selection and its resulting deployment performance**.
