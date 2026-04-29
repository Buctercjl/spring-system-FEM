import numpy as np
a=1 #常数
# --------------------------
# 1. 节点坐标 (x, y)
# --------------------------
nodes = np.array([
    [0, 0],# 节点0
    [2*a, 0],# 节点1
    [a, -a]   # 节点2
])

# --------------------------
# 2. 单元连接
# --------------------------
elements = [
    (0, 1),
    (0, 2),
    (2, 1)
]

# --------------------------
# 3. 材料参数
# --------------------------
E = 1
A = 1

n_nodes = len(nodes)
ndof = 2 * n_nodes   # 每节点2自由度

# --------------------------
# 4. 初始化刚度矩阵
# --------------------------
K = np.zeros((ndof, ndof))

# --------------------------
# 5. 组装刚度矩阵
# --------------------------
for i, j in elements:
    xi, yi = nodes[i]
    xj, yj = nodes[j]

    L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
    c = (xj - xi) / L
    s = (yj - yi) / L

    k = (E*A/L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])

    dof = [2*i, 2*i+1, 2*j, 2*j+1]

    for a in range(4):
        for b in range(4):
            K[dof[a], dof[b]] += k[a, b]

# --------------------------
# 6. 外力向量
# --------------------------
F = np.zeros(ndof)
F[2*1 + 1] = -1   # 节点1竖直向下1N

# --------------------------
# 7. 边界条件
# --------------------------
fixed_dofs = [0, 1, 2*2+1]  
# 节点0固定(x,y)，节点2固定y方向

free_dofs = list(set(range(ndof)) - set(fixed_dofs))

# --------------------------
# 8. 求解位移
# --------------------------
K_ff = K[np.ix_(free_dofs, free_dofs)]
F_f = F[free_dofs]

U = np.zeros(ndof)
U[free_dofs] = np.linalg.solve(K_ff, F_f)

# --------------------------
# 9. 输出位移
# --------------------------
print("节点位移：")
for i in range(n_nodes):
    ux = U[2*i]
    uy = U[2*i+1]
    print(f"节点{i}: ux={ux:.6e}, uy={uy:.6e}")

# --------------------------
# 10. 计算单元内力
# --------------------------
print("\n单元轴力：")
for idx, (i, j) in enumerate(elements):
    xi, yi = nodes[i]
    xj, yj = nodes[j]

    L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
    c = (xj - xi) / L
    s = (yj - yi) / L

    dof = [2*i, 2*i+1, 2*j, 2*j+1]
    u_e = U[dof]

    # 局部轴向变形
    delta = np.array([-c, -s, c, s]) @ u_e

    N = (E*A/L) * delta

    print(f"单元{idx} 轴力 = {N:.2f} N")