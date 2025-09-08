import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# 建立兩組資料：有幼蟲（實驗組）與無幼蟲（對照組）
with_larvae = np.array([
    0.571428571, 0, 0.244897959, 0.038167939, 0.238297872, 0.203285421, 
    0.246458924, 0.277492292, 0.206489676, 0.303231939, 0.183301344, 0.254493851, 
    0.129737609, 0.21819788, 0.081677704, 0.134340223, 0.037197232, 0.114825581, 
    0.038338658, 0.078674948, 0.017421603, 0.054744526
])
without_larvae = np.array([
    0, 0.010752688, 0, 0.015785861, 0.021276596, 0.018564356,
    0.137184116, 0.08456486, 0.125766871, 0.092560554, 0.087254063, 0.108674929,
    0.034674064, 0.115151515, 0.018715225, 0.077564637, 0.008245877, 0.029275809,
    0.006963788, 0.039140811, 0, 0.042875158
])

# 定義 ROPE 區間
rope_interval = [-0.01, 0.01]

# 建立 Bayesian model
with pm.Model() as model:
    mu1 = pm.Normal("mu1", mu=0, sigma=1)
    mu2 = pm.Normal("mu2", mu=0, sigma=1)
    sigma1 = pm.HalfNormal("sigma1", sigma=1)
    sigma2 = pm.HalfNormal("sigma2", sigma=1)

    group1 = pm.Normal("group1", mu=mu1, sigma=sigma1, observed=with_larvae)
    group2 = pm.Normal("group2", mu=mu2, sigma=sigma2, observed=without_larvae)

    diff_means = pm.Deterministic("diff_means", mu1 - mu2)

    trace = pm.sample(2000, return_inferencedata=True, cores=1, random_seed=42)

# 後驗分布與 ROPE 可視化
az.plot_posterior(trace, var_names=["diff_means"], rope=rope_interval)
plt.title("Posterior of the Difference in Means (With vs. Without Larvae)")
plt.show()