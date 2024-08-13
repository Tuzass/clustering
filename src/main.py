from functions import *
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import time

p = 2
metric = None
if p == 1:
    metric = 'manhattan'
elif p == 2:
    metric = 'euclidean'

folder_name = None
data, true_labels, k = readUciFile(folder_name)
print (f"The data contains {len(data)} instances, each with {len(data[0])} attributes, distributed among {k} classes")

initial_time = time.time()
if K_ENABLE_TABLE_GENERATION:
    distance_table = generateDistanceTable(data, p)
    table_time = round(time.time() - initial_time, K_DECIMAL_PLACES)
    print (f"Generating the distance table took {table_time} seconds")
else:
    distance_table = readUciTable(folder_name, p)
    table_time = round(time.time() - initial_time, K_DECIMAL_PLACES)
    print (f"Retrieving the distance table took {table_time} seconds")

if K_ENABLE_TABLE_GENERATION and K_ENABLE_WRITING:
    writeUciTable(distance_table, folder_name, p)

max_distance = getMaxDistance(distance_table)
print (f"Performing {K_NUMBER_OF_TESTS} tests, p = {p}")

result_file = open(folder_name + r'/results.csv', mode='w')
result_file.write('algorithm,test-number,p,search-depth,max-radius,run-time,rand-index,silhouette-coefficient')

# furthest first tests
for i in range(K_NUMBER_OF_TESTS):
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

print ('\nFurthest-First tests done')

# binary search tests
for d in range(2, 7):
    for i in range(K_NUMBER_OF_TESTS):
        results = ['bs', str(i + 1), str(p), str(d)]
        initial_time = time.time()
        centers = binarySearch(distance_table, 0, max_distance, depth=d, k=k)
        max_radius = getMaxRadius(centers, distance_table)
        final_time = time.time()
        rand = adjusted_rand_score(true_labels, found_labels)
        silhouette = silhouette_score(data, found_labels, metric=metric)
        results.extend([str(max_radius), str(final_time - initial_time), str(rand), str(silhouette)])
        result_file.write('\n' + ','.join(results))
    
    print (f'Binary-Search with depth {d} tests done')

# k-means tests
kmeans = KMeans(n_clusters=k)
for i in range(K_NUMBER_OF_TESTS):
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

print ('K-Means tests done')

result_file.write('\n')
result_file.close()

analyzeResults(folder_name)
