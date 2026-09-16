# laplace-ts

A [Laplace](https://github.com/mlatinov/laplace) library of time series building blocks for Stan — the classical ARIMA family, seasonal variants, exponential smoothing, GARCH volatility, smooth-transition regime switching, and count autoregression. Every model ships a mean recursion, a log density, a simulator, and a forecaster. Import it into any `.laplace` model and call it with namespaced calls (`ts::function_name(...)`).

Like all Laplace libraries, `ts` compiles down to plain, readable Stan functions. Nothing about how you use it hides what actually ends up in your `.stan` file.

## How it's organised

[#how-its-organised](#how-its-organised)

Every model in this library is a recursion over time, and the library gives you that recursion in four forms:

1. **A primary** computes the model's driving quantity from the data — the conditional mean $\mu_t$ for the linear models, the conditional standard deviation $\sigma_t$ for GARCH, the conditional intensity $\lambda_t$ for INGARCH. Use it when you want the quantity itself, for a custom likelihood or for `transformed parameters`.
2. **An `_lpdf` / `_lpmf`** runs the same recursion and scores the observed series. This is what goes in `model`.
3. **An `_rng`** runs the recursion forward from explicit seed values, drawing instead of scoring. For prior predictive checks and simulation-based calibration.
4. **A `_forecast_rng`** replays the recursion over the observed series to recover its internal state (residuals, intensities, level and seasonal states), then continues $H$ steps. For `generated quantities`.

`_rng` and `_forecast_rng` are not the same function with different arguments. `_rng` starts cold; `_forecast_rng` inherits everything the data implies. For a pure AR model the two bodies coincide, but for anything carrying a residual or state history they do not.

Every model also has a **covariate overload**: the same function with `matrix X, vector beta` appended after `sigma`. There is no separate `arimax` — you add the regressors to the model you already have:

```
// same recursion, with a regression on the mean
target += ts::arma_lpdf(y | alpha, phi, theta, sigma, X, beta);
```

Note this is ARMAX, not regression-with-ARMA-errors: the regressors sit inside the autoregressive loop, so `beta` is not the long-run effect of `X` on `y`. If you want the clean effect, model the residual structure separately.

## Data-side helpers

[#data-side-helpers](#data-side-helpers)

These are pure data transformations. Run them once in `transformed data` when their inputs are data, or inline when they aren't.

| Function | Returns | What it does |
| --- | --- | --- |
| `lag_design(p, y)` | `matrix[T-p, p]` | Design matrix whose row $n$ holds $y_{t-1},\dots,y_{t-p}$ for $t = p + n$ |
| `lag_design(p, y, L)` | `matrix[T-L, p]` | Same, but rows start at $t = L+1$. Needs $L \geq p$ |
| `seasonal_lag_design(P, s, y)` | `matrix[T-Ps, P]` | Seasonal lags $y_{t-s},\dots,y_{t-Ps}$ |
| `seasonal_lag_design(P, s, y, L)` | `matrix[T-L, P]` | Same, rows starting at $t = L+1$. Needs $L \geq Ps$ |
| `difference(y, d)` | `vector[T-d]` | $\nabla^d y$, ordinary differencing |
| `seasonal_difference(y, s)` | `vector[T-s]` | $\nabla_s y_t = y_t - y_{t-s}$ |
| `seasonal_difference(y, s, D)` | `vector[T-Ds]` | $\nabla_s^D y$ |
| `undifference(y_obs, delta_f, d)` | `vector[H]` | Integrates a forecast of $d$-th differences back onto the level scale |
| `unseasonal_difference(y_obs, delta_f, s, D)` | `vector[H]` | Same for seasonal differences |

The `L` overloads exist because `sar` adds an ordinary lag matrix to a seasonal one, and those two have different natural starting rows. Build both with `L = max(p, P * s)` or they won't line up. The two `un*` helpers are called for you inside `arima_forecast_rng` and `sarima_forecast_rng`; you only need them directly if you're forecasting by hand.

## Linear Gaussian mean models

[#linear-gaussian-mean-models](#linear-gaussian-mean-models)

All of these put $y_t \sim \mathcal{N}(\mu_t, \sigma)$ and differ only in how $\mu_t$ is built. $\varepsilon_t = y_t - \mu_t$ throughout, computed rather than sampled, and set to zero before the series starts.

| Model | Arguments after `y_obs` | $\mu_t$ |
| --- | --- | --- |
| `ar` | `alpha, phi, X_lag` | $\alpha + \sum_{i=1}^{p}\phi_i y_{t-i}$ |
| `ma` | `alpha, theta` | $\alpha + \sum_{j=1}^{q}\theta_j\varepsilon_{t-j}$ |
| `arma` | `alpha, phi, theta` | $\alpha + \sum_i \phi_i y_{t-i} + \sum_j \theta_j\varepsilon_{t-j}$ |
| `sar` | `alpha, phi, Phi, x_lag, X_slag` | $\alpha + \sum_i \phi_i y_{t-i} + \sum_k \Phi_k y_{t-ks}$ |
| `sma` | `alpha, theta, Theta, s` | $\alpha + \sum_j \theta_j\varepsilon_{t-j} + \sum_k \Theta_k\varepsilon_{t-ks}$ |
| `sarma` | `alpha, phi, Phi, theta, Theta, s` | see below |
| `arima` | `alpha, d, phi, theta` | the ARMA recursion applied to $z = \nabla^d y$ |
| `sarima` | `alpha, phi, Phi, theta, Theta, d, D, s` | the SARMA recursion applied to $z = \nabla^d\nabla_s^D y$ |

The seasonal ARMA mean, written out:

$$
\mu_t = \alpha + \sum_{i=1}^{p}\phi_i y_{t-i} + \sum_{k=1}^{P}\Phi_k y_{t-ks} + \sum_{j=1}^{q}\theta_j\varepsilon_{t-j} + \sum_{n=1}^{Q}\Theta_n\varepsilon_{t-ns}
$$

This is the **additive** seasonal form, not the classical multiplicative $\phi(B)\Phi(B^s)$ one. The multiplicative operator expands to the same lags plus cross-terms at $i + ks$ with coefficients $-\phi_i\Phi_k$, which constrains parameters the data rarely justify. Additive is simpler to read, simpler to prior, and what this library ships.

`ar` and `sar` take design matrices rather than the raw series, so their mean is a single matrix–vector product with no loop. Everything else loops, because the residual history has to be built one step at a time.

### Differencing and what it does to the likelihood

[#differencing-and-what-it-does-to-the-likelihood](#differencing-and-what-it-does-to-the-likelihood)

`arima_lpdf` and `sarima_lpdf` return the density of the **differenced** series:

$$
\log p(z) = \sum_t \log \mathcal{N}(z_t \mid \mu_t, \sigma), \qquad z = \nabla^d \nabla_s^D y
$$

Two models with different $d$ are therefore scoring different data, and their `lp__` and `loo` values are not comparable. Pick $d$ from the series, not from model comparison.

`arima_forecast_rng` and `sarima_forecast_rng` hide this: they difference, forecast on the differenced scale, and integrate back, so what you get out is on the level scale.

## Beyond the linear Gaussian mean

[#beyond-the-linear-gaussian-mean](#beyond-the-linear-gaussian-mean)

| Model | Arguments after `y_obs` | What it models |
| --- | --- | --- |
| `exponential_smoothing` | `alpha, beta, gamma, l0, b0, s0, m` | Additive Holt–Winters: level, trend and seasonal, all updated from the data |
| `garch` | `omega, alpha, beta, mu` | Conditional variance, given any conditional mean |
| `star` | `alpha1, alpha2, phi1, phi2, c, kappa, d, scale` | Two AR regimes with a smooth transition between them |
| `ingarch` | `omega, alpha, beta` | Log-linear count autoregression |

**Holt–Winters** is a deterministic filter: given the data and the parameters there are no latent states at all, only $(\alpha, \beta, \gamma)$, the seeds, and $\sigma$. That makes it the cheapest useful forecaster here.

$$
\hat{y}_t = \ell_{t-1} + b_{t-1} + s_{t-m}
$$

$$
\ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})
$$

$$
b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}
$$

$$
s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-m}
$$

The seasonal update uses $y_t - \ell_t$; some references use $y_t - \ell_{t-1} - b_{t-1}$ instead. Both appear in the literature and they behave similarly, but they are not identical, so don't expect an exact match against another package.

**GARCH** takes the conditional mean as a vector argument, so it composes with any mean model in this library or one you wrote yourself:

$$
\varepsilon_t = y_t - \mu_t, \qquad \sigma_t^2 = \omega + a\,\varepsilon_{t-1}^2 + b\,\sigma_{t-1}^2, \qquad \sigma_1^2 = \frac{\omega}{1 - a - b}
$$

$\omega$ is a variance floor, not the long-run variance; the long-run variance is $\omega / (1 - a - b)$, which is also how the recursion is initialised.

**STAR** is regime switching without any latent states, because the regime weight depends on an *observed* lag:

$$
G_t = \mathrm{logit}^{-1}\!\left(\kappa\,\frac{y_{t-d} - c}{\text{scale}}\right), \qquad
\mu_t = (1 - G_t)\,m^{(1)}_t + G_t\,m^{(2)}_t
$$

where $m^{(r)}_t = \alpha_r + \sum_j \phi^{(r)}_j y_{t-j}$ is regime $r$'s own AR mean. It's differentiable everywhere, which the hard-threshold TAR version is not, so it samples properly under HMC.

**INGARCH** is written on the log scale so the intensity can't go negative:

$$
\eta_t = \omega + \sum_{p}\alpha_p\log(1 + y_{t-p}) + \sum_{q}\beta_q\log\lambda_{t-q}, \qquad \lambda_t = e^{\eta_t}, \qquad y_t \sim \text{Poisson}(\lambda_t)
$$

The covariate overload adds $x_t^\top\beta_c$ to $\eta_t$, inside the link.

## The four forms, by name

[#the-four-forms-by-name](#the-four-forms-by-name)

For a model called `<name>`:

| Form | Shape | Where it goes |
| --- | --- | --- |
| `<name>(...)` | Returns $\mu_t$ (or $\sigma_t$, $\lambda_t$, $\hat{y}_t$) | `transformed parameters`, or `model` with your own likelihood |
| `<name>_lpdf(y \| ...)` | Returns a real log density | `model`, via `target +=` |
| `<name>_rng(H, ...)` | Returns a simulated series of length $H$ | `transformed data` or `generated quantities` |
| `<name>_forecast_rng(y_obs, H, ...)` | Returns $H$ draws for $t = T+1,\dots,T+H$ | `generated quantities` |

The ARMA-family primaries return `array[2] vector`: position 1 is $\mu_t$, position 2 is $\varepsilon_t$. INGARCH uses `_lpmf` rather than `_lpdf` and returns `array[] int` from its simulators.

Every function carries `@brief`, `@param`, `@return`, `@math`, and (where useful) `@example` documentation, so you can read it from the terminal without leaving your model:

```
laplace doc ts::sarima_lpdf
```

## Installation

[#installation](#installation)

`ts` is distributed as a git-hosted Laplace library — there's no published registry entry yet, so it's added by pointing `laplace` (or `cmdlaplacer`, if you're working from R) directly at the repository. The package lives in the repository's `laplace/` subdirectory, so pass it as the subdir.

### Via the `laplace` CLI

[#via-the-laplace-cli](#via-the-laplace-cli)

From inside a Laplace project (a directory with its own `laplace.toml`):

```
laplace add ts --git https://github.com/mlatinov/laplace-ts --tag 0.1.0 --subdir laplace
```

### Via R (`cmdlaplacer`)

[#via-r-cmdlaplacer](#via-r-cmdlaplacer)

```
library(cmdlaplacer)

laplace_install_git(
  "ts",
  "https://github.com/mlatinov/laplace-ts",
  tag = "0.1.0",
  subdir = "laplace"
)
```

Either way, this pins the dependency in your project's `laplace.toml`/`laplace.lock` at tag `0.1.0`. Check the [tags](https://github.com/mlatinov/laplace-ts/tags) for newer versions as they become available.

## Usage

[#usage](#usage)

Import the library in a `library { }` block and call its functions with the `ts::` namespace prefix.

### AR(p) with covariates and a forecast

[#arp-with-covariates-and-a-forecast](#arp-with-covariates-and-a-forecast)

The design matrix is built once in `transformed data`, so the AR mean costs one matrix–vector product per gradient. Note that `y` is scored from $t = p+1$ onwards: the first `p` observations are conditioned on, never modelled.

```
library {
    import ts
}

data {
  int<lower=1> T;
  int<lower=1> p;
  vector[T] y;
  int<lower=0> K;                       // number of covariates
  matrix[T, K] X_all;                   // covariates over the whole series
  int<lower=1> H;                       // forecast horizon
  matrix[H, K] X_future;                // covariates over the horizon
}

transformed data {
  int N = T - p;
  matrix[N, p] X_lag = ts::lag_design(p, y);
  vector[N] y_obs    = y[(p + 1):T];
  matrix[N, K] X     = X_all[(p + 1):T];
}

parameters {
  real alpha;
  vector[p] phi;
  vector[K] beta;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 2);
  phi   ~ normal(0, 0.5);              // shrinks toward stationarity without enforcing it
  beta  ~ normal(0, 1);
  sigma ~ exponential(1);

  target += ts::ar_lpdf(y_obs | alpha, phi, X_lag, sigma, X, beta);
}

generated quantities {
  vector[H] y_forecast = ts::ar_forecast_rng(y, H, alpha, phi, sigma, X_future, beta);
}
```

Drop `X`/`beta` from both calls for the plain AR model — the overload without covariates has the same name.

### SARIMA on monthly data

[#sarima-on-monthly-data](#sarima-on-monthly-data)

One ordinary difference for the trend, one seasonal difference for the yearly cycle, and short AR/MA terms at both scales. The forecaster handles the integration back to levels.

```
library {
    import ts
}

data {
  int<lower=1> T;
  vector[T] y;
  int<lower=1> H;
}

transformed data {
  int s = 12;                           // monthly data, yearly season
  int d = 1;                            // one ordinary difference
  int D = 1;                            // one seasonal difference
}

parameters {
  real alpha;                           // drift, on the differenced scale
  vector[1] phi;                        // AR(1)
  vector[1] Phi;                        // seasonal AR(1)
  vector[1] theta;                      // MA(1)
  vector[1] Theta;                      // seasonal MA(1)
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 1);
  phi   ~ normal(0, 0.5);
  Phi   ~ normal(0, 0.5);
  theta ~ normal(0, 0.5);
  Theta ~ normal(0, 0.5);
  sigma ~ exponential(1);

  target += ts::sarima_lpdf(y | alpha, phi, Phi, theta, Theta, d, D, s, sigma);
}

generated quantities {
  vector[H] y_forecast =
    ts::sarima_forecast_rng(y, H, alpha, phi, Phi, theta, Theta, d, D, s, sigma);
}
```

`alpha` here is a drift on the twice-differenced scale, so it is *not* the level of the series. With $d = 1$ it is the average period-on-period change; with $d = 2$ it is the average change in that change, and a prior centred anywhere but zero is usually a mistake.

### ARMA mean with GARCH volatility

[#arma-mean-with-garch-volatility](#arma-mean-with-garch-volatility)

`garch` takes the conditional mean as a vector, so any mean model composes with it. Here the primary supplies $\mu_t$ and GARCH supplies $\sigma_t$, and the likelihood is written by hand because neither function alone is the whole model.

```
library {
    import ts
}

data {
  int<lower=1> T;
  vector[T] y;                          // e.g. log returns
  int<lower=1> H;
}

parameters {
  real alpha_mean;
  vector[1] phi;
  vector[1] theta;

  real<lower=0> omega;                  // variance floor
  real<lower=0, upper=1> a;             // reactivity
  real<lower=0, upper=1> b;             // persistence
}

transformed parameters {
  array[2] vector[T] mean_eps = ts::arma(y, alpha_mean, phi, theta);
  vector[T] mu = mean_eps[1];
}

model {
  alpha_mean ~ normal(0, 1);
  phi        ~ normal(0, 0.5);
  theta      ~ normal(0, 0.5);
  omega      ~ exponential(10);
  a          ~ beta(2, 8);
  b          ~ beta(8, 2);

  target += ts::garch_lpdf(y | omega, a, b, mu);
}

generated quantities {
  vector[H] mu_future = ts::arma_forecast_rng(y, H, alpha_mean, phi, theta, sqrt(omega));
  vector[H] y_forecast = ts::garch_forecast_rng(y, H, omega, a, b, mu, mu_future);
}
```

The stationarity constraint $a + b < 1$ is not expressible as a box constraint on two parameters. Either declare `b` as `real<lower=0, upper=1-a>` in your own model, or put a `simplex[3]` over $(\omega\text{-share}, a, b)$ if you want it enforced exactly. The `beta(2,8)` / `beta(8,2)` priors above make violations unlikely rather than impossible, which is the usual applied compromise.

One caveat on that forecast block: `arma_forecast_rng` draws its own innovations at a constant `sqrt(omega)`, and `garch_forecast_rng` then draws again with the time-varying $\sigma_t$. That gives a mean path and a volatility path that are individually right but not driven by the same innovations. It is fine for a quick predictive interval and wrong if you care about the joint path. A single call that runs both recursions off one innovation stream isn't in 0.1.0; until it is, write the combined loop yourself in `generated quantities`, or use a constant mean, which is the usual setup for returns anyway.

### From R

[#from-r](#from-r)

With `cmdlaplacer`, the `.laplace` file compiles straight to a `cmdstanr` model, and the generated `.stan` file stays on disk next to it:

```
library(cmdlaplacer)

mod <- laplace_model("sarima.laplace")
fit <- mod$sample(data = list(T = length(y), y = y, H = 24))

fit$draws("y_forecast")
```

## Things to know

[#things-to-know](#things-to-know)

- **Use `target +=`, not `~`.** Laplace rewrites `ts::func(` calls, so write `target += ts::arma_lpdf(y | ...)`. The `y ~ ts::arma(...)` form won't resolve.
- **Conditional likelihood only.** The first observations are conditioned on rather than modelled, which is what almost all applied Stan code does. The exact likelihood needs the stationary covariance of the initial state, which means solving a Lyapunov equation, and that isn't here.
- **Stationarity and invertibility are not enforced.** There is no Monahan transform, no partial-autocorrelation reparameterization. Put informative priors on `phi` and `theta` — `normal(0, 0.5)` is a reasonable default — and check the posterior rather than the parameterization. If the sampler wanders into an explosive region your predictions will tell you loudly.
- **`arima_lpdf` and `sarima_lpdf` score the differenced series.** `lp__` and `loo` are not comparable across models with different `d` or `D`.
- **`sar` needs matching design matrices.** Build both with `lag_design(p, y, L)` and `seasonal_lag_design(P, s, y, L)` at the same `L = max(p, P * s)`, or the two products have different row counts.
- **Covariate overloads index `X` on the modelled scale.** For `ar`/`sar` that means the lag-trimmed rows; for `arima`/`sarima` it means the *differenced* series, so `X` has `T - D*s - d` rows. Difference your regressors too, or the alignment is silently wrong.
- **`_rng` starts cold, `_forecast_rng` inherits the data.** Use `_rng` for prior predictive checks and SBC, `_forecast_rng` for anything conditioned on a fit. Mixing them up gives forecasts that ignore where the series actually is.
- **MA forecasts go flat after `q` steps.** The residual inputs run out, so the mean settles at `alpha` and only the innovation variance remains. This is correct, not a bug.
- **`_rng` functions are restricted by Stan.** They can only be called in `transformed data` or `generated quantities`.
- **Holt–Winters seeds must sum to zero.** `s0` has length `m` and holds $s_{-m+1},\dots,s_0$. If they don't sum to zero, the seasonal component and the level are not separately identified and the level will drift to absorb the difference.
- **STAR's `d` is data, never a parameter.** The delay indexes into the series, so it can't be sampled. Grid over it and compare, or fix it. `kappa` is divided by `scale` to make it dimensionless — pass the standard deviation of `y`. When `kappa` is large the transition is nearly a step and `c` becomes weakly identified.
- **INGARCH intensities can overflow.** `poisson_rng` errors if $\lambda$ exceeds $2^{30}$, which a log-linear recursion can reach during warmup. Keep `omega` modestly prior-constrained and `beta` well below 1.
- **These recursions are sequential.** MA-type models build the residual history one step at a time, so they cost $O(T)$ non-vectorised operations per gradient evaluation. That's fine for thousands of points, slow for millions. The AR and SAR paths are the vectorised exception.
- **No state-space models in 0.1.0.** Local level, local linear trend, stochastic seasonality, dynamic regression and stochastic volatility all need either a Kalman filter (to marginalise) or a per-timepoint parameter vector (to sample directly). Planned, but not here yet. Hidden Markov regime switching is likewise absent; `star` covers the regime-switching use case without the forward algorithm.

## License

[#license](#license)

See [LICENSE](https://github.com/mlatinov/laplace-ts/blob/main/LICENSE).