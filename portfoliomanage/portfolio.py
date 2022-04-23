#!/usr/bin/python3
 

 
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import math
 
# 下記データは https://zog.jp/2332.html より
 
# 銘柄名
index = ['日本株式', '先進国株式', '新興国株式', '米国国債', '国内REIT', '米国REIT', '金']
# リターン
ret = np.array( [0.050, 0.055,  0.065,  0.025,  0.055,  0.050,  0.025])
# リスク
risk = np.array([0.175, 0.190,  0.240,  0.100,  0.190,  0.180,  0.160])
# 相関係数 (上三角のみ入力)
cor = np.array([[ 1.00,  0.70,   0.65,  -0.50,   0.60,   0.50,   0.05],
                [    0,  1.00,   0.85,  -0.40,   0.50,   0.75,   0.15],
                [    0,     0,   1.00,  -0.20,   0.45,   0.65,   0.30],
                [    0,     0,      0,   1.00,  -0.06,  -0.01,   0.00],
                [    0,     0,      0,      0,   1.00,   0.40,   0.05],
                [    0,     0,      0,      0,      0,   1.00,   0.05],
                [    0,     0,      0,      0,      0,      0,   1.00]])

 
ret = pd.Series(ret, index=index)
print("== Return ==\n%s\n" % ret)
 
risk = pd.Series(risk, index=index)
print("== Risk ==\n%s\n" % risk)
 
cor = np.triu(cor, k=1) + np.triu(cor).T
print("== Correlation matrix ==\n%s\n" % cor)
 
cov = pd.DataFrame(
    data = np.dot(np.diag(risk),np.dot(cor,np.diag(risk))))
print("== Covariance matrix ==\n%s\n" % cov)
 
ef = EfficientFrontier(ret, cov)
 
print("== Max Sharpe ==")
w = ef.max_sharpe()
perf = ef.portfolio_performance(True)
print("")
 
# print("== Min Volatility==")
# w = ef.min_volatility()
# perf = ef.portfolio_performance(True)
# print("")
 
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
 
tret = np.arange(perf[0], np.amax(ret) + 0.001, 0.001)
res_ret = []
res_risk = []
res_text = []
res_weight = {}
for key in ret.keys():
    res_weight[key] = []
 
for r in tret:
    w = ef.efficient_return(r)
    pref = ef.portfolio_performance()
    w = ef.clean_weights()
    print("Risk:%f, Ret:%f, Sharpe:%f %s" % (pref[1], pref[0], pref[2], w))
    res_risk += [pref[1] * 100.0]
    res_ret += [pref[0] * 100.0]
    for k, v in w.items():
        res_weight[k] += [v * 100.0]
 
data = [
    go.Scatter(
        x = res_risk,
        y = res_ret,
        hovertemplate = 'Ret: %{y:.1f}%'
                        'Risk: %{x:.1f}%',
        name = "リターン",
        mode = 'lines+markers',
        showlegend = False,
        yaxis = 'y1',
    )]
 
for k, v in res_weight.items():
    data += [dict(
        x = res_risk,
        y = v,
        mode = 'lines',
        name = k,
        hovertemplate = '%{y:.1f}%',
        stackgroup = 'weight',
        yaxis = 'y2',
    )]
 
axis_template = dict(
    zeroline = True,
    showgrid = True,
    rangemode = 'tozero',
    tickformat = ".1f",
    hoverformat = ".1f",
)
layout = go.Layout(
    xaxis = {
        **axis_template,
        'title': 'リスク [%]'
    },
    yaxis = {
        **axis_template,
        'side': 'left',
        'title': 'リターン [%]',
        'range': [0, math.ceil(max(res_ret))]
    },
    yaxis2 = dict(
        title = 'ウエイト [%]',
        side = 'right',
        overlaying = 'y',
        showgrid = False,
        range = [0,100]
    ),
    legend = dict(x = 0.01, y = 1.0),
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig,
                    filename='efficient_frontier.html',
                    show_link=False,
                    config={
                        "displayModeBar": False,
                    }
)