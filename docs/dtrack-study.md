# The Hilbert-Space Surrogate: When and Why It Beats (or Loses to) GP, RF, and TPE

**Mehdi Ghasemi** — SKSurrogate project, D-track study (2026-10-25)

> **Status:** standalone research note (D3 deliverable). Raw data and full regime
> tables in `docs/benchmarks/dtrack.md`; reproducible from
> `python -m benchmarks.dtrack.d1_approx` and `python -m benchmarks.dtrack.d2_compare`.
> Raw results archived at DOI [10.5281/zenodo.23652613](https://doi.org/10.5281/zenodo.23652613) (Zenodo, 2026-10-25).

---

## Abstract

`SKSurrogate` ships a custom response-surface regressor built on Hilbert-space
techniques (`HilbertRegressor`). This study characterizes *when and why* that
regressor beats or loses to the standard alternatives — a Gaussian process (GP),
a random forest (RF), and Optuna's TPE — across a controlled grid of synthetic
objective families, dimensions, sample sizes, and noise levels.

The headline result is a clean **regime map**:

- **Hilbert (polynomial basis, deg=3) is exact on low-degree polynomial
  objectives** — RMSE $\approx 0$ even at $n=10$ — and is **noise-robust**.
- **GP wins on clean, smooth, non-polynomial objectives** (Gaussian-bump and
  rational families at large $n$) but **collapses under noise** (the stock RBF GP
  has no noise term and interpolates the noise).
- **RF is the most robust generalist**: it wins the trigonometric family outright
  and most noisy regimes, at the cost of never being exact on polynomial targets.
- **Hilbert's weakness is its polynomial basis**: it fails on trigonometric
  objectives (RMSE $0.7$–$11$) and is **numerically unstable at small $n$ in
  higher dimension** (a singular Gram matrix: 2.2% of all D1 fits failed, every
  one at $d=3$, $\deg=3$, $n=10$).

The practical takeaway for surrogate-guided search: the Hilbert regressor is the
right tool when the objective is (near-)polynomial or the evaluations are noisy;
GP is best on clean smooth objectives; RF is the safe default when the objective
shape is unknown.

---

## 1. The method

`HilbertRegressor` (implemented in `SKSurrogate/NpyProximation.py`) approximates
a function $f$ by its **orthogonal projection onto a finite-dimensional subspace
of $L^2(X, \mu)$**, where $X \subset \mathbb{R}^d$ is the search domain and $\mu$
is a measure supported on the training points.

Concretely, given training points $\{(x_i, y_i)\}_{i=1}^n$:

1. A **discrete measure** $\mu = \sum_i \delta_{x_i}$ is placed on the training
   points, so the inner product is
   $\langle g, h \rangle_\mu = \sum_i g(x_i) h(x_i)$.
2. A **basis** $\{b_1, \dots, b_N\}$ is chosen — by default the monomials of
   degree at most $\deg$ in $d$ variables, so
   $N = \binom{d + \deg}{d}$.
3. The basis is **orthogonalized** against $\mu$ via the Gram matrix
   $G_{ij} = \langle b_i, b_j \rangle_\mu$ (a Gram–Schmidt-type construction using
   minors of $G$).
4. The approximation is the truncated orthogonal series
   $\hat f = \sum_{j=1}^N \langle f, b_j \rangle_\mu \, \tilde b_j$,
   where $\{\tilde b_j\}$ is the orthonormalized basis.

Because the measure is discrete, the projection is a **least-squares fit in the
span of the basis** — but computed through the orthogonal series rather than a
normal-equations solve. Two consequences follow immediately and are confirmed
empirically below:

- **If $f$ lies in the span of the basis** (e.g. a polynomial of degree $\le
  \deg$), the projection is *exact* regardless of $n$ — provided the Gram matrix
  is nonsingular.
- **The Gram matrix is singular when $n < N$** (fewer points than basis
  functions), and near-singular when the points are poorly spread — the source of
  the small-$n$ instability.

The default is $\deg = 3$; a Fourier basis is available via `FunctionBasis.Fourier`
but is not the default.

---

## 2. Experimental setup

### 2.1 Synthetic objectives (closed-form, on $[-1,1]^d$)

| family | form | why |
|--------|------|-----|
| `poly` | low-degree polynomial (deg 3) | in the Hilbert basis span → should be exact |
| `trig` | $\sum_i \sin(2\pi x_i) + \tfrac12 \cos(2\pi x_i)$ | smooth but *not* polynomial → Hilbert's weakness |
| `mixed` | $e^{-\|x\|^2}(1 + \tfrac12 \sum_i x_i)$ | smooth non-polynomial (Gaussian bump) |
| `rational` | $1/(1 + \|x\|^2)$ | non-polynomial, pole-free |

Dimensions $d \in \{1,2,3\}$. All responses are $O(1)$ so RMSE and noise levels
are comparable across families.

### 2.2 D1 — approximation behavior

- **Convergence:** RMSE on a 2000-point test grid vs $n \in \{10,20,50,100,200,500\}$,
  for $\deg \in \{1,2,3\}$; log–log slope estimates the rate.
- **Small-$n$ stability:** mean ± std RMSE over 20 repeated draws at
  $n \in \{10,20,30\}$; fit failures (singular Gram) recorded.
- **Noise:** RMSE at $n=100$ as Gaussian noise $\sigma \in \{0, 0.01, 0.05, 0.1\}$
  is added to the evaluations.

### 2.3 D2 — head-to-head surrogate comparison

Three surrogate *regressors* — `HilbertRegressor(deg=3)`, stock
`GaussianProcessRegressor` (RBF), and `RandomForestRegressor` (100 trees) — are
fit on the *same* training samples and scored by RMSE on the same test grid, over
the grid (family × dim × $n \in \{20,100,500\}$ × $\sigma \in \{0, 0.05\}$ × 10
repeats). TPE is not a point-predictor regressor (it is a density-ratio sampler),
so it is not in this table; it enters the study as a search baseline in §5.

---

## 3. Results

### 3.1 D1 — convergence, stability, noise

**Convergence.** Hilbert(deg=3) is *exact* on the polynomial family at every $n$
(RMSE $\approx 0$; the log–log slope is then meaningless, dominated by
floating-point noise). On the non-polynomial families the RMSE is essentially
**flat in $n$** (slopes $\approx 0$ to $-0.5$): the error is set by the *basis
truncation*, not the sample size. This is the central D1 finding — **Hilbert's
convergence is basis-limited, not sample-limited.**

**Small-$n$ stability.** On the polynomial family, Hilbert(deg=3) is exact even at
$n=10$. But at $d=3$, $\deg=3$ the basis has $N = \binom{6}{3} = 20$ functions,
and with only $n=10$ points the Gram matrix is singular: **2.2% of all D1 fits
failed, and every failure is at $d=3$, $\deg=3$, $n=10$** (21 per family). The
trigonometric family also shows large variance at small $n$ (e.g. RMSE $30 \pm
104$ at $d=2$, $n=10$, $\deg=3$) — near-singular Gram matrices amplify noise.

**Noise.** Hilbert(deg=3) degrades *gracefully* with noise on the polynomial
family (RMSE $0.005 \to 0.025$ at $d=1$ as $\sigma: 0 \to 0.1$) and stays flat on
mixed/rational. On trig it is already poor and noise is irrelevant.

### 3.2 D2 — regime map

Winner counts by objective family (18 cells each):

| family | Hilbert | GP | RF |
|--------|---------|----|----|
| `poly` | **14** | 2 | 2 |
| `mixed` | 2 | **10** | 6 |
| `rational` | 3 | 5 | **10** |
| `trig` | 0 | 1 | **17** |

Reading the map:

- **Hilbert dominates the polynomial family** (14/18), including under noise and
  at small $n$ — its exactness on in-span objectives is a genuine advantage.
- **GP dominates clean smooth objectives** (`mixed` 10/18, and the clean
  large-$n$ rational cells) but **collapses under noise**: the stock RBF GP
  (default `alpha=1e-10`) interpolates the noise, so its RMSE jumps from
  $\approx 0$ (clean) to $0.6$–$2.9$ (noisy) — worse than random.
- **RF dominates trigonometric and noisy regimes** (17/18 on `trig`; most noisy
  cells across families) — it is the most robust generalist, at the cost of never
  being exact on polynomial targets.

---

## 4. When and why Hilbert beats or loses

**Hilbert wins** when the objective is (near-)polynomial of degree $\le \deg$, or
when evaluations are noisy and the objective is smooth. Its orthogonal-series
construction gives an *exact* fit on in-span objectives and a principled
least-squares projection otherwise, with no noise-parameter tuning.

**Hilbert loses** when the objective is far from the polynomial span —
trigonometric, oscillatory, or high-frequency — because the polynomial basis
cannot represent it (the error is basis-truncation, not sample-limited), and when
$n$ is small relative to the basis size in higher dimension, where the Gram
matrix becomes singular.

**The practical rule:** the Hilbert regressor is a *specialist* — excellent on
polynomial-like, noisy objectives; a poor generalist. GP is a clean-smooth
specialist that needs a noise term to survive noisy evaluations. RF is the robust
generalist. For surrogate-guided search, this argues for **surrogate selection
based on the objective's suspected smoothness/spectrum**, or an ensemble that
falls back to RF when the Hilbert fit is unstable.

---

## 5. Relation to the B4 benchmark (TPE and real datasets)

The B4 equal-budget benchmark (real OpenML datasets, `docs/benchmarks.md`) found
the stock RBF-GP surface *edges out* the default cubic Hilbert regressor at
$B=30$ and *decisively* at $B=60$ ($p \approx 10^{-4}$), with Optuna-TPE the
strongest baseline. The D-track explains *why*: real HPO objectives (cross-validated
accuracy over discrete hyperparameter grids) are **not polynomial** in the
parameters, so the Hilbert regressor's polynomial basis is at a structural
disadvantage — exactly the regime where GP/RF win in the synthetic study. The two
studies are consistent: **Hilbert's advantage is confined to polynomial-like
objectives, which real HPO landscapes rarely are.**

---

## 6. Limitations

- Synthetic objectives are smooth and low-dimensional ($d \le 3$); real HPO
  spaces are higher-dimensional and discrete.
- The GP is the *stock* RBF GP (no noise term, no hyperparameter tuning); a
  tuned GP with a noise term would be more noise-robust.
- TPE is compared only as a search baseline (B4), not as a point-predictor in the
  D2 regressor table.
- The polynomial-basis limitation suggests a **Fourier-basis Hilbert variant**
  (`FunctionBasis.Fourier`) as a natural follow-up; it is not evaluated here.

---

## 7. Reproducibility

- **Code:** `benchmarks/dtrack/` (`objectives.py`, `d1_approx.py`, `d2_compare.py`,
  `analyze.py`).
- **Run:** `python -m benchmarks.dtrack.d1_approx` and
  `python -m benchmarks.dtrack.d2_compare`, then `python -m benchmarks.dtrack.analyze`.
- **Raw data:** `benchmarks/results/dtrack/` (git-ignored; archived to Zenodo).
- **Report + figures:** `docs/benchmarks/dtrack.md`,
  `docs/benchmarks/figures/dtrack_d1_convergence.png`,
  `docs/benchmarks/figures/dtrack_d2_regime_map.png`.
- **Environment:** Python 3, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4.
