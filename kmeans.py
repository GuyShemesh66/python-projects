import math
import sys
 

def dist_l2(v1, v2):
    sum = 0
    for i in range(0, len(v1)):
        sum += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(sum)

def kmeans(*args):
    if len(args) == 2:
        it = 200
        cluster_num = args[0]
        input_file_name = args[1]
    elif len(args) == 3:
        cluster_num = args[0]
        it = args[1]
        input_file_name = args[2]
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
        
    if type(it) != int or it >= 1000 or it < 1:
        print("Invalid maximum iteration!") 
        return 1
    
    if type(cluster_num) != int or cluster_num < 1:
        print("Invalid number of clusters!")   
        return 1

    input_file = open(input_file_name)
    epsilon = 0.001
    input_vectors = []
    for line in input_file.readlines():
        vector = line.split(",")
        for i in range(len(vector)):
            vector[i] = float(vector[i])
        input_vectors.append(vector)
    means_list = []
    if cluster_num >= len(input_vectors):
        print("Invalid number of clusters!")
        return 1
    for i in range(cluster_num):
        means_list.append(input_vectors[i])

    i = 0
    while i < it:
        clusters = []
        for _ in range(cluster_num):
            clusters.append([])

        for vector in input_vectors:
            min_dist = -1
            close_cluster = -1

            for cluster_index in range(len(means_list)):
                dist = dist_l2(means_list[cluster_index], vector)

                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    close_cluster = cluster_index

            clusters[close_cluster].append(vector)

        converge = True

        for cluster_index in range(len(means_list)):
            new_mean = [0] * len(input_vectors[0])
            size = len(clusters[cluster_index])

            for vector in clusters[cluster_index]:
                new_mean = [a + b for a, b in zip(new_mean, vector)]

            new_mean = [a / size for a in new_mean]

            if dist_l2(new_mean, means_list[cluster_index]) >= epsilon:
                converge = False

            means_list[cluster_index] = new_mean

        if converge:
            break
        
        i += 1
    for a in range(0,cluster_num):
        for b in range (0,len(input_vectors[0])-1):
                num = "%.4f" % means_list[a][b]
                print(str(num) + ",", end="")  
        
        print("%.4f" % means_list[a][len(input_vectors[0])-1])
    return 0

def main():
    args = sys.argv
    if len(args) == 3:
        kmeans(args[1], args[2])
    elif len(args) == 4:
        kmeans(args[1], args[2],args[3])


if __name__ == "__main__":
    main()
