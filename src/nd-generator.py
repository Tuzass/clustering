import numpy as np
import matplotlib.pyplot as plt

# if enabled, the sets in datasets/nd-generated will be overwritten
K_ENABLE_WRITING = False

centers_per_set = [[[0, 0], [5, 5], [10, 0], [-2, -5]],
                   [[0, -2], [2, 3], [7, 2]],
                   [[-4, 7], [6, -5], [3, 0], [-3, 2], [0, -4]],
                   [[0, 0], [2, -1], [4, 3], [-2, 2]],
                   [[9, 7], [5, 2], [9, 1]],
                   [[-1, -5], [4, 0], [3, -3], [2, -1], [-1, 3]],
                   [[0, 3], [4, 4], [-10, -2], [-6, 1]],
                   [[-3, -4], [-5, 2], [-2, 0]],
                   [[-2, -2], [3, 3], [4, 3], [5, -4], [2, 0], [1, -2], [-3, 2]],
                   [[2, 2], [2, -2], [-2, 2], [-2, -2]]]

stddevs_per_set = [[0.3, 0.4, 0.4, 0.5],
                   [0.8, 0.5, 0.8],
                   [0.75, 1.5, 1, 2, 1.25],
                   [1, 1, 0.65, 0.35],
                   [2, 1.5, 2],
                   [0.8, 1.35, 1.2, 1.75, 1],
                   [0.7, 1.8, 1, 2],
                   [2.4, 1.9, 1.4],
                   [1, 1.15, 1.45, 1.2, 1.4, 1.1, 1.25],
                   [2.5, 2.5, 2.5, 2.5]]

numpoints_per_set = [[100, 225, 150, 120],
                     [100, 150, 100],
                     [200, 100, 100, 150, 125],
                     [130, 145, 220, 80],
                     [170, 110, 200],
                     [270, 110, 150, 170, 210],
                     [225, 125, 175, 100],
                     [190, 115, 200],
                     [110, 125, 85, 90, 60, 120, 90],
                     [200, 125, 150, 200]]

for i in range(len(centers_per_set)):
    if K_ENABLE_WRITING:
        file = open(rf'datasets/nd-generated/set-{i + 1}/data.csv', mode='w')
    
    centers = centers_per_set[i]
    std_devs = stddevs_per_set[i]
    num_points = numpoints_per_set[i]

    for j in range(len(centers)):
        cluster_data = np.random.multivariate_normal(centers[j], np.diag([std_devs[j], std_devs[j]]), num_points[j])

        if not K_ENABLE_WRITING:
            continue

        for point in cluster_data:
            file.write(f'{j},{float(point[0])},{float(point[1])}\n')
    
    if K_ENABLE_WRITING:
        file.close()
