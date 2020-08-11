Config = {
    'num_classes': 21,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    # 方差：个人理解：除以variance是对预测box和真实box的误差进行放大，从而增加loss，增加梯度，加快收敛速度。
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}