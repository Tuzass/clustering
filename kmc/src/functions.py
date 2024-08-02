import numpy as np
import random

# constant for Chebyshev distance
K_CHEBYSHEV = np.inf

# decimal places in output
K_DECIMAL_PLACES = 3

# k parameter for the k-means clustering algorithms
K_PARAMETER = 5

# number of tests performed for each algorithm
K_NUMBER_OF_TESTS = 30

# depth of recursion in the binary search version of the algorithm
K_BINARY_SEARCH_DEPTH = 10

# enables and disables output of tests
K_TEST_OUTPUT_ENABLED = False

# reads points from file and returns a list of them in (x, y) format
def readPoints(file):
    points = []

    for line in file:
        try:
            space_index = line.find(" ")
            x_coord = float(line[:space_index])
            y_coord = float(line[space_index + 1:])
            points.append((x_coord, y_coord))
        except:
            return "Invalid format!"
    
    return points

# prints the given table, formated
# distances are rounded to 3 decimal places
def printTable(table):
    pad_size = 2
    n = len(table)
    row_number_pad = len(str(n - 1))

    str_table = []
    for row in table:
        str_row = [str(round(d, 3)) for d in row]
        str_table.append(str_row)
    
    max_length = 0
    for str_row in str_table:
        length = max([len(d) for d in str_row])
        max_length = max(max_length, length)
    
    for i in range(n):
        if i == 0:
            print ('\n' + (pad_size + row_number_pad + max_length - len(str(i))) * ' ' + str(i), end='')
        
        else:
            print ((pad_size + max_length - len(str(i))) * ' ' + str(i), end='')
    
    print ('\n')
    for i, str_row in enumerate(str_table):
        str_row = [' ' * (max_length - len(d)) + d for d in str_row]
        print (' ' * (row_number_pad - len(str(i))) + str(i) + ' ' * pad_size + (' ' * pad_size).join(str_row))

# returns the Minkowski distance or order p between points a and b
# for Chebyshev distance, set p = K_CHEBYSHEV
def minkowskiDistance(a, b, p):
    a = np.array(a)
    b = np.array(b)

    if p == K_CHEBYSHEV:
        distance = np.max(np.abs(a - b))
    
    else:
        result = np.sum(np.power(np.abs(a - b), p))
        distance = np.power(result, 1 / p)
    
    return distance

# returns a table containing the Minkowski distance of order p between any two points in the given list
def distanceTable(points, p):
    table = []
    length = len(points)

    for i in range(length):
        row = []
        a = points[i]

        for j in range(length):
            b = points[j]

            if i == j:
                row.append(0)

            elif i > j:
                row.append(table[j][i])

            else:
                d = minkowskiDistance(a, b, p)
                row.append(d)
        
        table.append(row)

    return table

# returns the longest distance in the given table
def getMaxDistance(table):
    longest = - np.inf

    for row in table:
        longest_in_row = max(row)
        if longest_in_row > longest:
            longest = longest_in_row

    return longest

# returns the greatest distance from a point to its closest center
def getMaxRadius(centers, table):
    max_distance = - np.inf
    number_of_points = len(table)
    points = list(range(number_of_points))

    for i in points:
        closest_distance = np.inf

        for j in centers:
            distance = table[i][j]
            if distance < closest_distance:
                closest_distance = distance
        
        if closest_distance > max_distance:
            max_distance = closest_distance

    return max_distance

# returns the set of centers
# uses the algorithm that chooses the furthest point to continue
def furthestFirstKMC(table, k=K_PARAMETER):
    number_of_points = len(table)
    points = list(range(number_of_points))

    if k >= number_of_points:
        return points

    first_point = random.choice(points)
    points.remove(first_point)
    centers = [first_point]

    while len(centers) < k:
        furthest = -1
        furthest_distance = - np.inf
        
        for i in points:
            closest_distance = np.inf
            for j in centers:
                distance = table[i][j]
                if distance < closest_distance:
                    closest_distance = distance
            if closest_distance > furthest_distance:
                furthest_distance = closest_distance
                furthest = i
        
        centers.append(furthest)
        points.remove(furthest)
    
    return centers

# returns the set of centers
# uses the algorithm that performs a binary search in the interval [lower_bound, upper_bound]
# to use log_file, just open 'log.txt' on main.py and pass it as an argument here
def binarySearchKMC(table, lower_bound, upper_bound, depth=K_BINARY_SEARCH_DEPTH, k=K_PARAMETER):
    # log_file.write(f"\n\n\nlower_bound = {round(lower_bound, K_DECIMAL_PLACES)}, upper_bound = {round(upper_bound, K_DECIMAL_PLACES)}, depth = {depth}")
    centers = []
    number_of_points = len(table)
    points = list(range(number_of_points))
    mid_point = lower_bound + (upper_bound - lower_bound) / 2
    # log_file.write(f"\nmid_point = {round(mid_point, K_DECIMAL_PLACES)}")

    while len(points) > 0:
        c = random.choice(points)
        centers.append(c)
        remaining = points.copy()

        for i in points:
            distance = table[c][i]
            if distance <= mid_point:
                remaining.remove(i)
        
        points = remaining.copy()

    if len(centers) <= k:
        # log_file.write(f"\nsuccess! centers = {centers}")
        if depth == 0:
            # log_file.write(f"\nreturning centers {centers}")
            return centers
        
        # log_file.write(f"\nbeginning search on lower half")
        lower_half = binarySearchKMC(table, lower_bound, mid_point, depth - 1, k)
        if lower_half != [None]:
            return lower_half
        
        # log_file.write(f"\n\n\nlower half did not produce results, returning centers {centers}")
        return centers
        
    else:
        # log_file.write(f"\nfailure! centers = {centers}")
        if depth == 0:
            # log_file.write("\nreturning [None]")
            return [None]
        
        # log_file.write(f"\nbeginning search on upper half")
        upper_half = binarySearchKMC(table, mid_point, upper_bound, depth - 1, k)
        if upper_half != [None]:
            return upper_half

    # log_file.write("\n\n\nupper half did not produce results, returning [None]")
    return [None]
