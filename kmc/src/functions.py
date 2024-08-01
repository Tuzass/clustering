import numpy as np

# constant for Chebyshev distance
K_CHEBYSHEV = np.inf

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
