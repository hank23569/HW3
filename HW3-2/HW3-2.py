import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import LinearSVC

# 定義高斯函數，用於計算第三維的值
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# Streamlit 介面
st.title("3D 方形分布和線性超平面展示")

# 生成資料
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# 計算每個點到原點的距離並分類
distances = np.sqrt(x1**2 + x2**2)
Y = np.where(distances < 4, 0, 1)

# 計算第三維度 x3
x3 = gaussian_function(x1, x2)

# 使用 LinearSVC 找到分隔超平面
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(max_iter=10000)
clf.fit(X, Y)
coef, intercept = clf.coef_[0], clf.intercept_

# 創建 3D 散點圖和超平面
fig = go.Figure()

# 繪製不同類別的點
fig.add_trace(go.Scatter3d(
    x=x1, y=x2, z=x3,
    mode='markers',
    marker=dict(size=5, color=Y, colorscale='Viridis'),
    name='Data Points'
))

# 繪製分隔超平面
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10), np.linspace(min(x2), max(x2), 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='Blues', opacity=0.5))

# 設定圖表的標題和軸標籤
fig.update_layout(
    scene=dict(
        xaxis_title='x1',
        yaxis_title='x2',
        zaxis_title='x3'
    ),
    title='3D Scatter Plot with Hyperplane'
)

# 顯示互動式圖表
st.plotly_chart(fig)
