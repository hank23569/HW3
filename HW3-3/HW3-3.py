import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import LinearSVC

# 定義高斯函數，用於計算第三維的值
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# Streamlit 介面
st.title("3D 方形分布和線性超平面展示")

# 拉桿讓使用者可以調整距離閾值，將下限設為 3.0
distance_threshold = st.slider("距離閾值", min_value=3.0, max_value=10.0, value=4.0, step=0.5)

# 生成方形分布的資料點
np.random.seed(0)
num_points = 600

# 方形的四個頂點座標
vertices = np.array([[5, 5], [5, -5], [-5, 5], [-5, -5]])

x1 = []
x2 = []
# 每個頂點周圍生成 num_points // 4 個點
for vx, vy in vertices:
    x1.extend(np.random.normal(vx, 1.5, num_points // 4))
    x2.extend(np.random.normal(vy, 1.5, num_points // 4))

x1 = np.array(x1)
x2 = np.array(x2)

# 計算每個點到原點的距離
distances = np.sqrt(x1**2 + x2**2)

# 根據距離閾值進行分類
Y = np.where(distances < distance_threshold, 0, 1)

# 計算第三維度 x3
x3 = gaussian_function(x1, x2)

# 使用 LinearSVC 找到分隔超平面
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# 創建互動式 3D 散點圖和超平面
fig = go.Figure()

# 繪製不同類別的點
fig.add_trace(go.Scatter3d(
    x=x1[Y == 0], y=x2[Y == 0], z=x3[Y == 0],
    mode='markers',
    marker=dict(size=5, color='blue', symbol='square'),
    name='Y=0'
))

fig.add_trace(go.Scatter3d(
    x=x1[Y == 1], y=x2[Y == 1], z=x3[Y == 1],
    mode='markers',
    marker=dict(size=5, color='red', symbol='square'),
    name='Y=1'
))

# 繪製分隔超平面
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10), np.linspace(min(x2), max(x2), 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='gray', opacity=0.5, showscale=False))

# 設定圖表的標題和軸標籤
fig.update_layout(
    scene=dict(
        xaxis_title='x1',
        yaxis_title='x2',
        zaxis_title='x3'
    ),
    title='3D Scatter Plot with Square Distribution and Separating Hyperplane'
)

# 顯示互動式圖表
st.plotly_chart(fig)
