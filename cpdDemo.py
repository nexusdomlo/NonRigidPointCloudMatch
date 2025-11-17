import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from pycpd import DeformableRegistration
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D # 导入3D绘图工具
import argparse, os

def visualize(iteration, error, X, Y, ax):
    """用于在配准过程中可视化3D结果的回调函数"""
    plt.cla()
    # 绘制三维散点图
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Source (Transformed)')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Target', alpha=0.4)
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:0.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.set_title('Deformable Registration (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
    plt.pause(0.1)

# create_fish_shape 函数不再需要，可以删除或保留
def create_fish_shape(num_points=100, scale=1.0):
    """创建一个鱼形的点云"""
    # ... (此函数内容未改变，但在新main函数中未被调用)
    body_t = np.linspace(-np.pi/2, np.pi/2, int(num_points * 0.7))
    body_x = scale * np.cos(body_t)
    body_y = scale * np.sin(2 * body_t) / 2
    tail_t = np.linspace(-np.pi/4, np.pi/4, int(num_points * 0.3))
    tail_x = -scale * (0.9 + np.abs(tail_t))
    tail_y = scale * 0.4 * np.tan(tail_t)
    x = np.concatenate([body_x, tail_x])
    y = np.concatenate([body_y, tail_y])
    return np.stack([x, y], axis=1)


def main():
    # 1. 从PCD文件加载源点云 (Source) 和目标点云 (Target)
    # !!! 请将下面的路径替换为您自己的PCD文件路径 !!!
    parser = argparse.ArgumentParser(description="register_with_prior: src tgt (supports .pcd/.ply/.npy)")
    parser.add_argument("src", help="source 点云 (.pcd/.ply/.npy)")
    parser.add_argument("tgt", help="target 点云 (.pcd/.ply/.npy) 或 已裁剪的 .pcd")
    #从命令行获取参数
    args = parser.parse_args()
    source_pcd_path = args.src
    target_pcd_path = args.tgt

    try:
        source_pcd = o3d.io.read_point_cloud(source_pcd_path)
        target_pcd = o3d.io.read_point_cloud(target_pcd_path)
    except Exception as e:
        print(f"错误：无法加载PCD文件。请检查文件路径是否正确。")
        print(e)
        return

    # 新增：对点云进行体素下采样
    # voxel_size 的值越大，降采样的程度越高，点云越稀疏，计算速度越快。
    # 您需要根据点云的尺寸和密度来调整这个值。
    voxel_size = 0.05 # 示例值，假设点云坐标单位为米，则体素大小为 5cm
    source_pcd_down = source_pcd.voxel_down_sample(voxel_size)
    target_pcd_down = target_pcd.voxel_down_sample(voxel_size)

    print(f"源点云从 {len(source_pcd.points)} 个点降采样到 {len(source_pcd_down.points)} 个点。")
    print(f"目标点云从 {len(target_pcd.points)} 个点降采样到 {len(target_pcd_down.points)} 个点。")

    # 将降采样后的open3d点云格式转换为numpy数组
    X = np.asarray(source_pcd_down.points)
    Y = np.asarray(target_pcd_down.points)

    # 2. 设置并运行非刚性配准
    # 注意：对于您自己的数据，可能需要调整 alpha 和 beta 值以获得最佳效果
    # alpha: 控制形变平滑度。值越大，形变越趋向于刚性。
    # beta:  高斯核的宽度。值越大，点之间的相互影响范围越广。
    reg = DeformableRegistration(**{'X': X, 'Y': Y, 'alpha': 0.5, 'beta': 2.0})
    
    # 3. 创建3D可视化窗口
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') # 设置为3D投影

    # 注册回调函数以在每次迭代时进行可视化
    reg.register(callback=lambda **kwargs: visualize(ax=ax, **kwargs))
    
    # 显示最终结果
    print("开始配准... 关闭绘图窗口后程序将结束。")
    plt.show()

if __name__ == '__main__':
    main()