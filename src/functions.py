from constants import *
import random

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

# reads data from datasets in CSV format and returns it, along with the number of classes (clusters)
# its only parameter is folder name, so it reads the contents of folder_name/data.csv
def readDataset(folder_name):
    file = open(folder_name + r'/data.csv')
    data = []
    true_labels = []
    classes = []

    for line in file:
        remaining_line = line.strip()
        instance_class = None
        instance = []

        while ',' in remaining_line:
            comma_index = remaining_line.index(',')
            attribute = remaining_line[:comma_index]
            remaining_line = remaining_line[comma_index + 1:]

            if instance_class is None:
                instance_class = attribute

                if instance_class in classes:
                    class_index = classes.index(instance_class)
                    true_labels.append(class_index)
                else:
                    true_labels.append(len(classes))
                    classes.append(instance_class)

            else:
                if len(attribute) == 0:
                    attribute = 0
                attribute = float(attribute)
                instance.append(round(attribute, K_DECIMAL_PLACES))
        
        instance.append(float(remaining_line))
        data.append(tuple(instance))

    return data, true_labels, len(classes)

# due to some tables taking a few minutes to calculate, the tables were stored for testing
# some of these tables were larger than 100mb, not really feasible to maintain
def readUciTable(folder_name, p):
    file = open(folder_name + rf'/table-{p}.csv', mode='r')
    table = []

    for i, line in enumerate(file):
        remaining_line = line.strip()
        row = []

        for j in range(i):
            row.append(table[j][i])

        row.append(0)
        while ',' in remaining_line:
            comma_index = remaining_line.index(',')
            distance = float(remaining_line[:comma_index])
            remaining_line = remaining_line[comma_index + 1:]
            row.append(round(distance, K_DECIMAL_PLACES))
        
        row.append(float(remaining_line))
        table.append(row)
    
    row = []
    for i in range(len(table)):
        row.append(table[i][-1])
    row.append(0)
    table.append(row)
    
    file.close()
    return table

# stores the distance table, to avoid future calculations
def writeUciTable(table, folder_name, p):
    file = open(folder_name + rf'/table-{p}.csv', mode='w')
    number_of_points = len(table)

    for i in range(number_of_points - 1):
        row = []

        for j in range(i + 1, number_of_points):
            distance = round(table[i][j], K_DECIMAL_PLACES)
            row.append(str(distance))
        
        file.write(','.join(row) + '\n')
    
    file.close()

# prints the given table, formated
# the number of decimal places can be changed in the constants file
def printTable(table):
    pad_size = 2
    n = len(table)
    row_number_pad = len(str(n - 1))

    str_table = []
    for row in table:
        str_row = [str(round(d, K_DECIMAL_PLACES)) for d in row]
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
def generateDistanceTable(points, p):
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
                d = minkowskiDistance(a, b, p).item()
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
# only works for FF or BS, since they reuse the distance table that's already been calculated
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

# returns the greatest distance from a point to its closest center
# works for all three algorithms, but it recalculates all distances
def getKmMaxRadius(data, centers, p):
    max_distance = - np.inf

    for i in data:
        closest_distance = np.inf

        for j in centers:
            distance = minkowskiDistance(i, j, p)
            if distance < closest_distance:
                closest_distance = distance
        
        if closest_distance > max_distance:
            max_distance = closest_distance

    return max_distance

# returns the set of centers
# uses the algorithm that chooses the furthest point to continue
def furthestFirst(table, k):
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
def binarySearch(table, lower_bound, upper_bound, depth, k):
    centers = []
    number_of_points = len(table)
    points = list(range(number_of_points))
    mid_point = lower_bound + (upper_bound - lower_bound) / 2

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
        if depth == 0:
            return centers
        
        lower_half = binarySearch(table, lower_bound, mid_point, depth - 1, k)
        if lower_half != [None]:
            return lower_half
        
        return centers
        
    else:
        if depth == 0:
            return [None]
        
        upper_half = binarySearch(table, mid_point, upper_bound, depth - 1, k)
        if upper_half != [None]:
            return upper_half

    return [None]

# finds the labels (closest center index) for each point
def findLabels(table, centers):
    number_of_points = len(table)
    found_labels = []

    for i in range(number_of_points):
        distance_to_centers = [table[i][j] for j in centers]
        shortest_distance = min(distance_to_centers)
        closest_center = distance_to_centers.index(shortest_distance)
        found_labels.append(closest_center)
    
    return found_labels

# outputs the average of the tests stored in folder_name/results.csv
def analyzeResults(folder_name, p):
    result_file = open(folder_name + rf'/results-p{p}.csv', mode='r')
    radii = []
    times = []
    rands = []
    silhouettes = []
    algorithms = {'bs': 'Binary-Search', 'ff': 'Furthest-First', 'km': 'K-Means'}

    for i, line in enumerate(result_file):
        if i == 0:
            continue

        results = line.split(',')
        if results[1] == '1':
            radii = [float(results[4])]
            times = [float(results[5])]
            rands = [float(results[6])]
            silhouettes = [float(results[7])]
        
        elif results[1] != '30':
            radii.append(float(results[4]))
            times.append(float(results[5]))
            rands.append(float(results[6]))
            silhouettes.append(float(results[7]))
        
        else:
            print (f"\n{algorithms[results[0]]} Algorithm:")

            if results[0] == 'bs':
                depth = int(results[3])
                print (f"Search depth: {depth} / Interval size: {round(1 / (2 ** depth), K_DECIMAL_PLACES) * 100}%")

            print (f"   - Average radius: {round(sum(radii) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)}")
            print (f"   - Average adjusted Rand index: {round(sum(rands) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)}")
            print (f"   - Average Silhouette coefficient: {round(sum(silhouettes) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)}")
            print (f"   - Average runtime: {round(sum(times) / K_NUMBER_OF_TESTS, K_DECIMAL_PLACES)} seconds")
