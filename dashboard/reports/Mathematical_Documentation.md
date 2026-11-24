# Mathematical formulation and optimization notes

## 1. Forecasting problem and units

Let `p_m` be the minute-average household active power in kW. For a complete hour ending at t,

\[
e_t=\frac{1}{60}\sum_{m\in[t-1h,t)}p_m\quad\text{kWh}.
\]

At forecast origin t, predict `y_t=e_(t+1)` from `x_t`, constructed using observations through t. Calendar features describe the next interval `[t,t+1h)`. The information set is `F_t = sigma(e_s: s<=t, calendar)`, and the forecast must be measurable with respect to `F_t`. Complete observed targets are never filled. Short missing input intervals can be causally forward-filled.

Training-only transformations produce `z_j=(x_j-mu_j)/s_j` (for standard scaling) and `u=(y-mu_y)/s_y`. All models in a comparison optimize the same standardized-target objective and use the same fitted transformations. Results are inverted to kWh. Changing feature scaling is its own experiment because it changes optimization geometry and the interpretation of a fixed regularization coefficient.

## 2. Linear model, loss, gradient, and Hessian

Append a column of ones to form the design matrix `A in R^(n x (d+1))`. Let `w=(theta,b)` and `D=diag(1,...,1,0)` so the intercept is not regularized.

\[
\hat u=Aw,\quad J(w)=\frac1n\|Aw-u\|_2^2+\frac\alpha2w^TDw.
\]

\[
\nabla J(w)=\frac2n A^T(Aw-u)+\alpha Dw,
\qquad H=\nabla^2J(w)=\frac2n A^TA+\alpha D.
\]

For any vector v, `v^T H v=2||Av||²/n+alpha||D v||²>=0`; therefore the linear/ridge objective is convex. It is strictly convex exactly when the Hessian is positive definite. Correlated calendar and lag variables, constant indicators, or redundant rolling features can make the unregularized Hessian singular. A tiny negative computed eigenvalue near machine precision is roundoff, not proof of non-convexity.

For alpha>0, the non-intercept directions are regularized and the intercept column is nonzero, giving a positive-definite Hessian in this formulation. Ridge improves conditioning by raising eigenvalues, though the unpenalized intercept means the eigenvalues are **not simply all shifted by alpha**. Its optimum solves `(2A^TA/n + alpha D)w = 2A^T u/n`.

The smoothness constant is `L=lambda_max(H)`. With positive-definite H, `mu=lambda_min(H)>0` and the spectral condition number is `kappa=L/mu`. For singular H, the full condition number is infinite/undefined; the implementation records JSON null and also reports the positive-spectrum condition number with an explicit numerical rank tolerance `L*1e-10`.

Batch GD on a quadratic is stable for `0<eta<2/L`; `eta<=1/L` ensures descent. Along each eigenvector its error contracts by `1-eta*lambda_i`. For strongly convex smooth losses, appropriate constant-step GD has a linear rate bounded in terms of mu/L. A bound from the **full-data** Hessian does not guarantee stability for every stochastic sample Hessian. Outliers can make the maximum single-sample curvature much larger.

The central difference gradient check is

\[
g_j^{num}=\frac{J(w+\epsilon e_j)-J(w-\epsilon e_j)}{2\epsilon},\quad
error=\frac{\|g^{num}-g\|_2}{\max(1,\|g^{num}\|_2,\|g\|_2)}.
\]

Unit tests check both random weights and L2/intercept behavior. Finite differences of gradients verify the Hessian action.

## 3. Neural model and non-convexity

The fixed network is `f(z)=W3 tanh(W2 tanh(W1 z+b1)+b2)+b3`, with hidden widths 32 and 16. Its loss is mean standardized squared error plus `alpha/2` times the squared Frobenius norms of weight matrices. Biases are excluded, matching the linear regularization convention. Autograd differentiates the same objective used for monitoring.

Parameter products and nonlinear activations make the parameter-space loss non-convex. Stationary points can be minima or saddles; the linear-model condition-number guarantee does not transfer globally. A small gradient alone does not certify a global optimum. Initialization, batch noise, saturation, learning rate, and optimizer state can affect the reachable basin.

## 4. Implemented optimizer updates

Let `g_t` denote the mini-batch mean gradient **including the exact L2 gradient**, and let epsilon be `1e-8`. Every state is zero-initialized. Batch GD uses all n samples; SGD uses one; mini-batch methods use b samples. Each epoch shuffles the same seeded permutation within a model/seed comparison.

| Method | Update | Additional state |
|---|---|---|
| Batch / stochastic / mini-batch GD | `w_(t+1)=w_t-eta_t*g_t` | O(p) gradient storage |
| Momentum | `v_t=beta*v_(t-1)-eta_t*g_t`; `w_(t+1)=w_t+v_t` | O(p) velocity |
| NumPy Nesterov | `q_t=w_t+beta*v_(t-1)`; evaluate `g(q_t)`; update velocity and weights as above | O(p) velocity |
| RMSprop | `s_t=.99*s_(t-1)+.01*g_t²`; `w_(t+1)=w_t-eta_t*g_t/(sqrt(s_t)+epsilon)` | O(p) second moment |
| Adam | `m_t=.9*m_(t-1)+.1*g_t`; `s_t=.999*s_(t-1)+.001*g_t²`; bias-correct both, then `w-=eta*mhat/(sqrt(shat)+epsilon)` | O(p) two moments |
| L-BFGS | Build inverse-Hessian action from recent `(s_k,y_k)` curvature pairs using two-loop recursion; select a step by line search | O(m*p) for history m=10 |

For Adam, `mhat=m/(1-.9^t)` and `shat=s/(1-.999^t)`; the counter increments per optimizer update, not per epoch. RMSprop does not use bias correction. `momentum` controls Momentum/Nesterov; Adam's beta1/beta2 are fixed. The NumPy Nesterov class explicitly evaluates a lookahead gradient. PyTorch uses its standard momentum-buffer Nesterov parameterization, `buffer=beta*buffer+g`, `direction=g+beta*buffer`, then an update of the current parameters. This is an implementation convention difference across model families; comparisons are made **within** each family, not by assuming identical NumPy/PyTorch iterates. Learning-rate decay also interacts with the momentum-buffer convention.

The two schedules are constant `eta_e=eta_0` and epoch decay `eta_e=eta_0/(1+.05*(e-1))`. Fixed finite-epoch decay is a sensitivity setting; it is not a claimed asymptotic convergence theorem. L-BFGS experiments use line searches with no fixed learning-rate sweep.

## 5. Computational complexity and fairness

For a linear model with p parameters and b samples, a first-order update costs O(bp) and extra optimizer state O(p). A full epoch costs O(np), but sample-wise Python/Autograd overhead can dominate SGD. For the neural network replace bp by the forward/backward network computation. Monitoring full objectives/gradients adds O(np) per checkpoint and is counted separately.

L-BFGS's two-loop inverse-Hessian action costs O(mp) per accepted update, plus q objective/gradient evaluations costing O(qnp) for full-data closures and O(mp) storage. q is variable, so an accepted update cannot be equated to one gradient evaluation. No dense Hessian is stored by the optimizer. The small linear Hessian is computed only for offline analysis.

The primary training budget is example-gradient processing: `epochs*n` for first-order methods and at most that budget for L-BFGS closures. Extra monitoring is logged separately. A full-data gradient is one call processing n examples; a sample gradient is one call processing one. Consequently gradient-call counts alone do not measure equal work. Wall time includes monitoring but excludes dataset loading and final metric exports. The sklearn solver is a direct reference and is not subject to this iterative budget.

The first seed's identical-baseline curves address controlled optimizer behavior. The selected leaderboard answers a different question: performance after each available validation search. Demo search budgets are unequal, so selected results cannot establish intrinsic superiority of the update rules.

## 6. Evaluation and stability

`MSE=mean((y-yhat)²)`, `RMSE=sqrt(MSE)` in kWh, `MAE=mean(abs(y-yhat))` in kWh, and `R²=1-SSE/SST`. MSE is in kWh². R² can be negative. Train/validation optimization curves are in standardized-target units, not kWh²; a regularized training objective also includes the L2 term.

Stability summaries include the fraction of monitored training-loss increases and the standard deviation of the last five monitored losses. These are descriptive, checkpoint-frequency-dependent statistics, not theoretical stability proofs. The gradient norm is the full regularized training-objective gradient. A preset loss threshold of .35 is checked at checkpoints; missing threshold time means it was not reached, not zero time. The best validation checkpoint is returned, while final losses and final gradient norm describe the last finite checkpoint; `best_epoch` makes that distinction explicit.

Sample SD (`ddof=1`) across three seeds quantifies variability under these seeds, not a confidence interval. Equal initialization and seeded shuffles support paired within-family comparisons. Deterministic batch methods should have the same results across seeds in linear models; timing can still vary.

## 7. Benchmark theory

* Quadratic: `f=.5(x²+4y²)`, Hessian diag(1,4), unique minimum (0,0).
* Ill-conditioned quadratic: `f=.5(x²+100y²)`, condition number 100, stable GD step `<.02`.
* Rosenbrock: `f=(1-x)²+100(y-x²)²`, minimum (1,1), curved valley; non-convex globally.
* Himmelblau: `f=(x²+y-11)²+(x+y²-7)²`, multiple minima; starting point matters.

Contour colors use `log(1+f)` for readability, while 3D surfaces show raw f. Gradient arrows are normalized and do not show momentum or adaptive directions. On deterministic exact-gradient functions, stochastic and mini-batch GD coincide with GD: there is no data sampling noise to compare. Animations show the first selected trajectory. Divergent paths stop gracefully rather than drawing non-finite geometry.

## References

UCI dataset DOI: https://doi.org/10.24432/C58K54. Adam: https://arxiv.org/abs/1412.6980. SciPy L-BFGS-B: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html. PyTorch optimizers and implementation conventions: https://docs.pytorch.org/docs/stable/optim.html.
