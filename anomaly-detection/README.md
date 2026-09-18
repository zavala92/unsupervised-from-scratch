# Anomaly detection, from scratch

Six stages. You write the detector; the test suite says when it is right; each
stage then runs it on data engineered so that one specific thing goes wrong.

numpy and matplotlib only, with a hand-rolled test runner so there is nothing
to install.

```bash
python3 tests/run_tests.py            # 21 tests, all TODO to begin with
python3 stages/stage1_density_1d.py   # each stage prints a narrative and a figure
```

## The premise

You have `m` examples of something behaving normally and no useful examples of
it behaving badly, because bad examples are rare and expensive. So you model
`p(x)` from the normal data and flag anything whose density falls below a
threshold. That is the whole algorithm. Everything difficult about it sits in
four questions, one per group of stages:

1. What does `p(x)` look like, and is it normalised? (stage 1)
2. What shape of "normal" can a product of marginals express? (stage 2)
3. How do you choose the threshold and report a score honestly? (stage 3)
4. What breaks that is not the model's fault? (stages 4 to 6)

## Working through it

`anomdet/core.py` holds nine functions, every one a `raise
NotImplementedError`. Fill them in. An unwritten function reports as `TODO`
rather than `FAIL`, so the summary line is a progress bar:

```bash
python3 tests/run_tests.py --stage 1     # only the tests you need next
python3 tests/run_tests.py -v            # full tracebacks
```

To see a finished stage before writing any code:

```bash
ANOMDET_USE_REFERENCE=1 python3 stages/stage2_engines_2d.py
```

`reference/reference_impl.py` holds worked versions of the nine functions
above. Read it actively if you read it at all: cover the body, try to
reproduce the function, look only for the hint you need.

These nine functions are this project's own exercises, not anyone else's. None
of them shares a name or a signature with a graded assignment in any course.

## What you implement

| function | stage | purpose |
|---|---|---|
| `fit_independent` | 1 | per-feature maximum-likelihood mean and variance |
| `independent_pdf` | 1 | `p(x)` as a product of independent Gaussians |
| `log_independent_pdf` | 2 | the same thing without ever forming the product |
| `precision_recall_f1` | 3 | the scoreboard, degenerate cases included |
| `tune_threshold` | 3 | choose epsilon by maximising F1 on validation data |
| `tune_threshold_log` | 3 | the same search on a grid uniform in `log p` |
| `fit_correlated` | 5 | full covariance by maximum likelihood |
| `log_correlated_pdf` | 5 | correlated density via Cholesky, no explicit inverse |
| `correlated_pdf` | 5 | the exponential of the above |

## The stages

**1. One feature, one density.** Fit `p(x)` to engine-heat measurements and
look at it. Check that it integrates to 1 by Gauss-Legendre quadrature to 14
digits; if the normalising constant is wrong, nothing downstream can be right.
Then threshold by hand and feel the trade-off before automating it.

**2. Heat and vibration.** The contours of `p` come out as axis-aligned
ellipses while the data cloud is tilted, which is the independence assumption
made visible. Two probe engines, each about 2 sigma out in both coordinates,
score within a factor of 3 of each other under the independent model, while a
correlation-aware distance separates them 4.0 to 2.0.

**3. Choosing epsilon.** A detector that flags nothing scores 93% accuracy on
this validation set and 99.9% in a realistic one, so accuracy is useless here.
F1, precision-recall curves, and the three-way train/validation/test split.
Also the first sign of a numerical point that grows teeth later: `p` spans 12
orders of magnitude, so a threshold grid spaced uniformly in `p` searches
almost nowhere and already loses F1 0.917 to 0.889 at n = 2.

**4. Feature shape and arithmetic.** Two failures with no statistical content.

*Part A.* A log-normal latency feature carrying no anomaly signal whatsoever
still drags precision from 1.00 to 0.61, because the fitted Gaussian places
11% of its mass on negative latency and consequently rates honest
slow-but-fine servers as bigger outliers than genuinely broken ones. Recall is
untouched. Bad feature shape costs precision, and a detector that cries wolf
gets switched off.

*Part B.* `p(x)` decays like `exp(-1.42n)` and hits the smallest positive
double at about n = 520. Past that every `p` is exactly `0.0`, `p < epsilon` is
true for everything, and the detector flags 100% of traffic while looking like
a tuning problem. Sums of logs have no such cliff.

**5. Full covariance.** Data where every anomaly lies within 2.1 sigma of the
mean in *every* coordinate while being jointly impossible, so no per-feature
rule can touch it. Test F1 goes from 0.217 to 1.000 on switching from diagonal
to full covariance. The density is implemented through the Cholesky factor,
with no `np.linalg.inv` and no `det`, and a test enforces it.

**6. Capstone.** Six features of synthetic iterative-solver telemetry
(condition number, iteration count, final residual, decay rate, setup time,
memory) with latent problem size and conditioning inducing real correlations,
and three planted failure modes. The harness scores recall per failure mode,
because the aggregate hides everything:

| model | test F1 | mode 1 | mode 2 | mode 3 |
|---|---|---|---|---|
| raw + diagonal | 0.345 | 0.62 | 0.00 | 0.00 |
| raw + full covariance | 0.345 | 0.62 | 0.00 | 0.00 |
| log + diagonal | 0.744 | 1.00 | 0.00 | 1.00 |
| log + full covariance | 0.958 | 1.00 | 1.00 | 0.88 |
| log + engineered residual + diagonal | 0.941 | 1.00 | 1.00 | 1.00 |

Target: overall F1 at least 0.90 and recall at least 0.85 on each mode. Each
mode needs a different one of the earlier lessons, so a mode stuck at 0.00
names the lesson still missing.

The first two rows are the point of the whole project. Adding off-diagonal
covariance to the raw features buys nothing, because the real dependence is
`iters ~ log(cond)`, which is not linear in the raw coordinates, and covariance
can only capture linear structure. The transform is a precondition for the
covariance model rather than an alternative to it. "What coordinates am I
looking at" is nearly always a more productive question than "which model
should I use".

Worked solution, including the engineered-feature route that reaches 1.00 on
all three modes with the cheap diagonal model:

```bash
python3 reference/stage6_worked.py
```

## Things worth trying next

* **Set epsilon by a false-alarm budget instead of F1.** Take the largest
  epsilon whose flag rate your inspection capacity can absorb, say 20 per day.
  This is what operations teams actually ask for, and it needs no labels.
* **Swap the Gaussian for a histogram or a kernel density estimate per
  feature.** The product structure does not care what you put in it. Where
  does this win, and what does it cost in the tail, where the decisions
  happen?
* **Shrinkage.** Fit a full covariance with `m` barely above `n` and watch the
  Cholesky fail, then regularise with `Sigma + lambda*I` and tune lambda on the
  validation set. How does the best lambda scale with `m/n`?
* **Multimodal normality.** Make the training set two well-separated clusters.
  A single Gaussian puts its peak in the empty space between them and calls
  both clusters anomalous. Fix it by clustering first, which is what the
  `kmeans/` directory is for.
* **Drift.** Shift the test distribution slightly and re-score without
  refitting. How much drift before the false-alarm rate doubles? This is the
  failure mode that kills deployed detectors.
* **Your own data.** The interface is `Dataset(X_train, X_val, y_val, X_test,
  y_test, feature_names, meta)` with all-normal training data.

## Layout

```
anomdet/core.py              the nine functions to write
anomdet/data.py              datasets, each engineered for one lesson
anomdet/plotting.py          figure helpers
tests/run_tests.py           21 tests, ordered by stage
stages/stage[1-6]_*.py       run these
reference/reference_impl.py   reference implementation
reference/stage6_worked.py worked capstone
figures/                     output
```
