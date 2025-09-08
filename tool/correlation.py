import numpy as np
from scipy.stats import pearsonr
# from sklearn.linear_model import LinearRegression

x = np.array([
    164.7, 353.729, 347.154, 199.442, 175.409, 156.606, 48.312, 0, 22.072, 43.938,
    62.952, 152.429, 137.981, 87.392, 71.103, 0, 58.954, 178.329, 182.997, 65.079,
    116.832, 61.392, 0, 0, 594.621, 651.665, 424.918, 240.025, 126.369, 27.907, 0, 
    3928.038, 1467.746, 1254.242, 1008.555, 429.196, 237.563, 3955.925, 2133.788, 
    1301.524, 977.458, 617.119, 372.788, 220.935, 3457.054
])

y = np.array([
    44.3, 161.2, 160.9, 87.7, 74.9, 42.5, 10.2, 0, 5.2, 33.7,
    47.7, 129.6, 82.7, 72.4, 17.5, 0, 5, 126.8, 81, 13.7,
    20.4, 28.1, 0, 0, 377.6, 697.3, 472, 241.7, 87.6, 10.1, 0, 
    9824.2, 4586.7, 3660.2, 1963.1, 600.9, 224.8, 10381.2, 
    6395.1, 2686.2, 1500, 795.4, 369.2 ,201.5, 5694.8
])
# corrcoe = np.corrcoef(x, y)
# corrcoe2 = pearsonr(x,y)
# print(corrcoe)
# print(corrcoe[0])

# ## 訓練與建構迴歸模型
# regressor = LinearRegression()
# regressor.fit(x, y)

# ## 計算出截距值與係數值
# w_0 = regressor.intercept_
# w_1 = regressor.coef_

# print('Interception : ', w_0)
# print('Coeficient : ', w_1)
# 計算「過原點」的相關係數
def correlation_through_origin(x, y):
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return numerator / denominator if denominator != 0 else np.nan

r_origin = correlation_through_origin(x, y)
print("Correlation through the origin:", r_origin)