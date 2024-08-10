from functions import *
from constants import *
import time

p = 2
folder_name = 'statlog-vehicle'
data, k = readUciFile(folder_name)
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

ff_radii = []
ff_times = []
bs_radii = []
bs_times = []

for i in range(K_NUMBER_OF_TESTS):
    initial_ff_time = time.time()
    ff_centers = furthestFirstKMC(distance_table, k)
    ff_max_radius = getMaxRadius(ff_centers, distance_table)
    final_ff_time = time.time()
    ff_radii.append(ff_max_radius)
    ff_times.append(final_ff_time - initial_ff_time)

    initial_bs_time = time.time()
    bs_centers = binarySearchKMC(distance_table, 0, max_distance, K_BINARY_SEARCH_DEPTH, k)
    bs_max_radius = getMaxRadius(bs_centers, distance_table)
    final_bs_time = time.time()
    bs_radii.append(bs_max_radius)
    bs_times.append(final_bs_time - initial_bs_time)

print (f"\nAfter {K_NUMBER_OF_TESTS} tests")
print (f"    - Furthest First had an average radius of {round(sum(ff_radii) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)}, with an average execution time of {round(sum(ff_times) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)} seconds")
print (f"    - Binary Search had an average radius of {round(sum(bs_radii) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)}, with an average execution time of {round(sum(bs_times) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)} seconds")

print (f"\nTotal runtime: {round(time.time() - initial_time, K_DECIMAL_PLACES)} seconds")
