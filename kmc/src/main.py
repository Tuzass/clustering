from functions import *
import numpy as np

points = []
with open("points.txt") as points_file:
    points = readPoints(points_file)

p1_distance_table = distanceTable(points, 1)
p2_distance_table = distanceTable(points, 2)
chebyshev_distance_table = distanceTable(points, K_CHEBYSHEV)

print ("\n\nDistance table for p = 1:")
printTable(p1_distance_table)

print ("\n\nDistance table for p = 2:")
printTable(p2_distance_table)

print ("\n\nChebyshev distance table:")
printTable(chebyshev_distance_table)

ff_max_radii = []
print ("\n\n" + "-" * 80 + "\n\n\nFurthest-first k-means clustering")
print (f"k = {K_PARAMETER}")

for i in range(K_NUMBER_OF_TESTS):
    centers = furthestFirstKMC(p2_distance_table)
    max_radius = getMaxRadius(centers, p2_distance_table)
    ff_max_radii.append(max_radius)
    if K_TEST_OUTPUT_ENABLED:
        print (f"\n\nTest {i + 1}\nCenters: {centers}\nMax radius: {round(max_radius, K_DECIMAL_PLACES)}")

average = round(sum(ff_max_radii) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)
print (f"\n\nNumber of tests: {K_NUMBER_OF_TESTS}\nAverage max radius: {average}")

bs_max_radii = []
max_distance = getMaxDistance(p2_distance_table)
print ("\n\n" + "-" * 80 + "\n\n\nBinary-search k-means clustering")
print (f"k = {K_PARAMETER}, binary search depth = {K_BINARY_SEARCH_DEPTH}")

for i in range(K_NUMBER_OF_TESTS):
    centers = binarySearchKMC(p2_distance_table, 0, max_distance)
    max_radius = getMaxRadius(centers, p2_distance_table)
    bs_max_radii.append(max_radius)
    if K_TEST_OUTPUT_ENABLED:
        print (f"\n\nTest {i + 1}\nCenters: {centers}\nMax radius: {round(max_radius, K_DECIMAL_PLACES)}")

average = round(sum(bs_max_radii) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)
print (f"\n\nNumber of tests: {K_NUMBER_OF_TESTS}\nAverage max radius: {average}")
print ("\n\n" + "-" * 80 + "\n\n")
