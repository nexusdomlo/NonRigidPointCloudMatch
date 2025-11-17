import numpy as np
import matplotlib.pyplot as plt
# from pycpd import DeformableRegistration # 不再使用pycpd
import probreg as pr # 导入probreg库
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import argparse
import time # 导入time模块以计算时间

def visualize(iteration, error, X, Y, ax):
    """用于在配准过程中可视化3D结果的回调函数"""
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', s=10, label='Source (Transformed)') # s参数调整点的大小
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', s=10, label='Target', alpha=0.4)
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:0.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.set_title('BCPD++ Registration (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
    plt.pause(0.1)

def main():
    parser = argparse.ArgumentParser(description="BCPD++ registration: src tgt (supports .pcd/.ply/.npy)")
    parser.add_argument("src", help="source 点云 (.pcd/.ply/.npy)")
    parser.add_argument("tgt", help="target 点云 (.pcd/.ply/.npy)")
    args = parser.parse_args()
    source_pcd_path = args.src
    target_pcd_path = args.tgt

    try:
        source_pcd = o3d.io.read_point_cloud(source_pcd_path)
        target_pcd = o3d.io.read_point_cloud(target_pcd_path)
    except Exception as e:
        print(f"错误：无法加载PCD文件。请检查文件路径是否正确。\n{e}")
        return

    voxel_size = 0.05
    source_pcd_down = source_pcd.voxel_down_sample(voxel_size)
    target_pcd_down = target_pcd.voxel_down_sample(voxel_size)

    print(f"源点云从 {len(source_pcd.points)} 个点降采样到 {len(source_pcd_down.points)} 个点。")
    print(f"目标点云从 {len(target_pcd.points)} 个点降采样到 {len(target_pcd_down.points)} 个点。")

    X = np.asarray(source_pcd_down.points)
    Y = np.asarray(target_pcd_down.points)

    # 2. 设置并运行BCPD++非刚性配准
    # 注意：BCPD++不需要手动设置alpha和beta
    # use_cuda=True 可以启用GPU加速
    cbs = [lambda T: visualize(T.iteration, T.err, T.TY, Y, ax)]
    
    # 创建3D可视化窗口
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print("开始BCPD++配准...")
    start_time = time.time()
    
    # 使用 probreg.bcpd 进行配准
    tf_param, _, _ = pr.bcpd(X, Y, max_iter=50, callbacks=cbs)
    
    end_time = time.time()
    print(f"配准完成，耗时: {end_time - start_time:.4f} 秒。")
    
    # 显示最终结果
    plt.show()

if __name__ == '__main__':
    main()