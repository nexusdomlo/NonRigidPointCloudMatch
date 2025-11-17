### 每个算法的介绍
Coherent Point Drift (CPD),将点云视为高斯混合模型，最大化概率似然,鲁棒性强，理论完备,计算量大，速度慢

### 简单测试CPD算法
```
python test.py 用于检测测试这个算法
```

### 使用cpd算法进行点云配准
```
python cpddemo.py src_path tgt_path 使用两个点云路径作为参数输入，然后运行一下
```

### 使用bcpd++算法进行点云配准
```
python bcpdplusDemo.py src_path tgt_path
```
