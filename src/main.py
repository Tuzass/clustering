from functions import *
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import time

p = None
metric = None
if p == 1:
    metric = 'manhattan'
elif p == 2:
    metric = 'euclidean'

# set the folder's name to retrieve the data stored in folder_name/data.csv
folder_name = None
data, true_labels, k = readDataset(folder_name)
if data is None:
    exit()

print (f"The data contains {len(data)} instances, each with {len(data[0])} attributes, distributed among {k} classes")

if K_ENABLE_CALCULATIONS:
    initial_time = time.time()
    distance_table = generateDistanceTable(data, p)
    table_time = round(time.time() - initial_time, K_DECIMAL_PLACES)
    print (f"Generating the distance table took {table_time} seconds")

    max_distance = getMaxDistance(distance_table)
    print (f"Performing {K_NUMBER_OF_TESTS} tests, p = {p}")

    result_file = open(folder_name + rf'/results-p{p}.csv', mode='w')
    result_file.write('algorithm,test-number,p,search-depth,max-radius,run-time,rand-index,silhouette-coefficient')

# furthest-first tests
for i in range(K_NUMBER_OF_TESTS):
    if not K_ENABLE_CALCULATIONS:
        break

    results = ['ff', str(i + 1), str(p), 'None']
    initial_time = time.time()
    centers = furthestFirst(distance_table, k)
    max_radius = getMaxRadius(centers, distance_table)
    final_time = time.time()
    found_labels = findLabels(distance_table, centers)
    rand = adjusted_rand_score(true_labels, found_labels)
    silhouette = silhouette_score(data, found_labels, metric=metric)
    results.extend([str(max_radius), str(final_time - initial_time), str(rand), str(silhouette)])
    result_file.write('\n' + ','.join(results))

if K_ENABLE_CALCULATIONS:
    print ('\nFurthest-First tests done')

# binary-search tests
for d in range(2, 7):
    if not K_ENABLE_CALCULATIONS:
        break

    for i in range(K_NUMBER_OF_TESTS):
        results = ['bs', str(i + 1), str(p), str(d)]
        initial_time = time.time()
        centers = binarySearch(distance_table, 0, max_distance, depth=d, k=k)
        
        while (len(centers) < k):
            point = random.randint(0, len(data) - 1)
            if point not in centers:
                centers.append(point)

        max_radius = getMaxRadius(centers, distance_table)
        final_time = time.time()
        found_labels = findLabels(distance_table, centers)
        rand = adjusted_rand_score(true_labels, found_labels)
        silhouette = silhouette_score(data, found_labels, metric=metric)
        results.extend([str(max_radius), str(final_time - initial_time), str(rand), str(silhouette)])
        result_file.write('\n' + ','.join(results))
    
    print (f'Binary-Search (depth {d}) tests done')

# k-means tests
kmeans = KMeans(n_clusters=k)
for i in range(K_NUMBER_OF_TESTS):
    if not K_ENABLE_CALCULATIONS:
        break

    results = ['km', str(i + 1), str(p), 'None']
    initial_time = time.time()
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    found_labels = kmeans.labels_
    max_radius = getKmMaxRadius(data, centers, p)
    final_time = time.time()
    rand = adjusted_rand_score(true_labels, found_labels)
    silhouette = silhouette_score(data, found_labels, metric=metric)
    results.extend([str(max_radius), str(final_time - initial_time), str(rand), str(silhouette)])
    result_file.write('\n' + ','.join(results))

if K_ENABLE_CALCULATIONS:
    print ('K-Means tests done')

    result_file.write('\n')
    result_file.close()
else:
    print (rf'Analyzing results stored in {folder_name}/results-p{p}.csv')

if K_ENABLE_ANALYSIS:
    analyzeResults(folder_name, p)
