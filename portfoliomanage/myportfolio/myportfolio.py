# https://kamiura.netlify.app/post/2020/0420-portfolio/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# input data, ファンドの海より
# http://guide.fund-no-umi.com/tools/aa.html
# -------------------------------------------------
r = np.array([1.00, 4.80, 3.50, 5.00, 9.25])  # 期待収益率
rho = np.array([[1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]])  # 相関係数
sigma = np.array([5.4, 22.15, 13.25, 19.59, 26.25])  # リスク（標準偏差）
# -------------------------------------------------


def ratio(n):  # n資産の割合を返す関数
    while True:
        rand_arr = np.zeros(n)  # 乱数用の配列
        rat_norm = np.zeros(n)  # 規格化後の投資比率

        for index, _ in enumerate(rand_arr):
            rand_arr[index] = np.random.rand()

        for index, _ in enumerate(rat_norm):
            rat_norm[index] = round(rand_arr[index] / np.sum(rand_arr), 2)

        diff = np.sum(rat_norm) - 1.0
        rat_norm[np.argmax(rat_norm)] -= diff  # なんとなく調整

        if np.sum(rat_norm) == 1.0:  # 合計が1.0になったらbreak
            break

    return rat_norm

# -------------------------------------------------


n = 5  # 資産の数

c = 0  # while文のカウンタ

arr = np.zeros(n + 2)  # データ格納用配列

while c < 10000:

    x = ratio(n)  # 資産の組み合わせを取得

    # 期待収益率:rp
    rp = 0
    for i in range(0, n):
        rp += r[i] * x[i]

    # 分散:sigmasp
    sigmasp = 0
    for i in range(0, n):
        for j in range(0, n):
            sigmasp += rho[i][j] * sigma[i] * sigma[j] * x[i] * x[j]

    sigmap = np.sqrt(sigmasp)  # 分散→標準偏差

    # 資産の組み合わせとrp, sigmapの値を持った配列aを作成
    a = np.insert(x, n, sigmap)
    a = np.insert(a, n, rp)

    # 配列arrに配列aをスタックしていく。
    arr = np.vstack([arr, a])

    c += 1

# 配列arrをDataFrame化. １行目は0が入っているので除く
df = pd.DataFrame(arr[1:], columns=['日本債券', '日本株式',
                                    '先進国債券', '先進国株式', '新興国株式', 'rp', 'sigmap'])


# DataFrameを条件で抽出
# 任意のrpにおいてsigmapが最小となる組み合わせ
minsig_loc = df.query('3.95<rp<4.05')['sigmap'].idxmin()
df_lsig = df[minsig_loc:minsig_loc + 1]

# 任意のsigmapにおいてrpが最大となる組み合わせ
maxrp_loc = df.query('6.95<sigmap<7.05')['rp'].idxmax()
df_mrp = df[maxrp_loc:maxrp_loc + 1]


# 表示とプロット
print(df_lsig)
print(df_mrp)

ax = plt.gca()

df.plot(x='sigmap', y='rp', kind='scatter', ax=ax)
df_lsig.plot(x='sigmap', y='rp', kind='scatter', color='red', ax=ax)
df_mrp.plot(x='sigmap', y='rp', kind='scatter', color='orange', ax=ax)

plt.show()

# 円グラフ
df_lsig = df_lsig.drop(['rp', 'sigmap'], axis=1)
label = ['j_bond', 'j_stock', 'adv_bond', 'adv_stock', 'dev_stock']
plt.pie(df_lsig, labels=label)