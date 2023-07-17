import math
import sys
import numpy as np
import pandas as pd
import mykmeanssp as km


def kmeans_pp(*args):
    if len(args) == 4:
        it = 300
        cluster_num = args[0]
        epsilon = args[1]
        file_1 = args[2]
        file_2 = args[3]
    elif len(args) == 5:
        cluster_num = args[0]
        it = args[1]
        epsilon = args[2]
        file_1 = args[3]
        file_2 = args[4]
    else:
        print("An Error Has Occurred")

    try:
        it = int(it)

    except:
        print("Invalid maximum iteration!")
        return 1

    try:
        cluster_num = int(cluster_num)
    except:
        print("Invalid number of clusters!")
        return 1
    
    epsilon = float(epsilon)

    if type(it) != int or it >= 1000 or it < 1:
        print("Invalid maximum iteration!")
        return 1

    if type(cluster_num) != int or cluster_num < 1:
        print("Invalid number of clusters!")
        return 1
    
    with open(file_1, 'r') as file:
        first_line = file.readlines()[0]
        point = [float(num) for num in first_line.split(',')]
        size_point1 = len(point)
        d = 2 * (len(point) - 1)
    with open(file_2, 'r') as file:
        first_line = file.readlines()[0]
        point = [float(num) for num in first_line.split(',')]
        size_point2 = len(point)


    data_1 = pd.read_csv(file_1, sep=',', header=None, names=[str(i) for i in range(size_point1)])
    data_2 = pd.read_csv(file_2, sep=',', header=None, names=[str(i) for i in range(size_point2)])
    merged_data = pd.merge(data_1, data_2, on='0', how='inner')

    merged_data = merged_data.sort_values(by='0')
    data = merged_data.values[:, 1:].astype(float)

    # Convert data to numpy array
    data_not_chosen = merged_data.values[:, 1:].astype(float)


    np.random.seed(0)
    # Choose the first centroid uniformly at random
    i = np.random.choice(len(data))
    x = data[i]
    centroids = [data[i]]

    # Remove the element from the list of what we haven't chosen yet
    index = np.where(np.all(data_not_chosen == x, 1))
    data_not_chosen = np.delete(data_not_chosen, index, 0)


    # Choose the remaining centroids
    for _ in range(cluster_num - 1):
    # Compute the distances between each data point and the nearest centroid
        m = len(data_not_chosen)
        all_dis = np.zeros(m, dtype=np.float64)
        probabilities = np.zeros(m, dtype=np.float64)
        for idx, x in enumerate(data_not_chosen):
            # Compute Euclidean distance between each data point from what we didn't choose and the centroid
            distance = np.inf  # reset distance to infinity at the beginning of each iteration
            for centroid in centroids:
                dis = np.sqrt(np.sum((x - centroid) ** 2))
                distance = np.minimum(distance, dis)
            all_dis[idx] = distance  # directly use idx instead of searching for index

    # Choose a new data point as a centroid based on the weighted probability distribution
        for i in range(m):
            probabilities[i] = all_dis[i] / np.sum(all_dis)
        j = np.random.choice(np.arange(m), p=probabilities)
        centroids.append(data_not_chosen[j])
        data_not_chosen = np.delete(data_not_chosen, j, 0)

    

    data = data.tolist()
    n = len(data)
    c = [subarray.tolist() for subarray in centroids]
    final_centroids = km.fit(c, data, it, epsilon, n, d, cluster_num)

    print(','.join(str(data.index(x)) for x in c))
    print('\n'.join(','.join('%.4f' % c for c in centroid) for centroid in final_centroids))

    return 0


def main():
    args = sys.argv
    if len(args) == 5:
        kmeans_pp(args[1], args[2], args[3], args[4])
    elif len(args) == 6:
        kmeans_pp(args[1], args[2], args[3], args[4], args[5])


if __name__ == "__main__":
    main()

