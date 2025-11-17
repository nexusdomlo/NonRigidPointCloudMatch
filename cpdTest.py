import numpy as np
import matplotlib.pyplot as plt
from pycpd import DeformableRegistration

def visualize(iteration, error, X, Y, ax):
    """用于在配准过程中可视化结果的回调函数"""
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Source (Transformed)')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Target', alpha=0.4)
    ax.text(0.87, 0.92, 'Iteration: {:d}\nError: {:0.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.set_title('Deformable Registration')
    plt.draw()
    plt.pause(0.1)

def create_fish_shape(num_points=100, scale=1.0):
    """创建一个鱼形的点云"""
    # 身体
    body_t = np.linspace(-np.pi/2, np.pi/2, int(num_points * 0.7))
    body_x = scale * np.cos(body_t)
    body_y = scale * np.sin(2 * body_t) / 2
    
    # 尾巴
    tail_t = np.linspace(-np.pi/4, np.pi/4, int(num_points * 0.3))
    tail_x = -scale * (0.9 + np.abs(tail_t))
    tail_y = scale * 0.4 * np.tan(tail_t)
    
    x = np.concatenate([body_x, tail_x])
    y = np.concatenate([body_y, tail_y])
    
    return np.stack([x, y], axis=1)

def main():
    # 1. 生成源点云 (Source) 和目标点云 (Target)
    # 源点云是一个标准的鱼形
    X = create_fish_shape(num_points=100, scale=1.0)

    # 目标点云是源点云经过非刚性变换（弯曲）后得到的
    Y = X.copy()
    Y[:, 1] = Y[:, 1] + 0.2 * np.sin(np.pi * Y[:, 0]) # 添加一个弯曲变形
    
    # 为了让问题更有挑战性，对目标点云进行旋转和平移
    angle = np.pi / 4
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    Y = np.dot(Y, rotation_matrix)
    Y += np.array([0.5, 0.5]) # 平移

    # 2. 设置并运行非刚性配准
    # alpha: 控制形变平滑度的正则化权重。值越大，形变越趋向于刚性。
    # beta:  高斯核的宽度。值越大，点之间的相互影响范围越广。
    reg = DeformableRegistration(**{'X': X, 'Y': Y, 'alpha': 0.5, 'beta': 2.0})
    
    # 创建可视化窗口
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    ax = fig.add_subplot(111)

    # 注册回调函数以在每次迭代时进行可视化
    reg.register(callback=lambda **kwargs: visualize(ax=ax, **kwargs))
    
    # 显示最终结果
    plt.show()

if __name__ == '__main__':
    main()