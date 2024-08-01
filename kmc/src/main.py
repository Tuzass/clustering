from functions import *
import numpy as np

points = []
with open("points.txt") as points_file:
    points = readPoints(points_file)

"""
p1_distance_table = distanceTable(points, 1)
p2_distance_table = distanceTable(points, 2)
chebyshev_distance_table = distanceTable(points, K_CHEBYSHEV)

print ("\n\nDistance table for p = 1:")
printTable(p1_distance_table)

print ("\n\nDistance table for p = 2:")
printTable(p2_distance_table)

print ("\n\nChebyshev distance table:")
printTable(chebyshev_distance_table)
"""
