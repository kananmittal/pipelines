# Pipeline Performance Analysis Report
Generated on: analysis_report.md

## 1. Descriptive Statistics (Hallucination Score: Lower is Better)
| Pipeline                             |   N (Samples) |   Mean Score |   Std Dev |   95% CI (Mean ±) |
|--------------------------------------|---------------|--------------|-----------|-------------------|
| Pipeline 2 (Consolidation)           |            40 |       0.13   |    0.2009 |            0.0642 |
| Pipeline 3 (Parallel)                |            40 |       0.145  |    0.0286 |            0.0091 |
| Pipeline 5 (Consolidation+Iterative) |            40 |       0.1812 |    0.2989 |            0.0956 |


## 2. Comparative Analysis (Paired/Independent T-Test)
Note: Negative Effect Size (Cohen's d) means the first pipeline is BETTER (lower score) if comparing Method A vs Method B.
| Comparison                                                                  |   T-Statistic |   P-Value | Significance   |   Effect Size (Cohen's d) |
|-----------------------------------------------------------------------------|---------------|-----------|----------------|---------------------------|
| Pipeline 3 (Parallel) vs Pipeline 2 (Consolidation) (Paired)                |         0.459 |    0.649  | ns             |                     0.105 |
| Pipeline 5 (Consolidation+Iterative) vs Pipeline 2 (Consolidation) (Paired) |         0.854 |    0.3985 | ns             |                     0.201 |
| Pipeline 5 (Consolidation+Iterative) vs Pipeline 3 (Parallel) (Paired)      |         0.757 |    0.4539 | ns             |                     0.171 |