import numpy as np

# constant for Chebyshev distance
K_CHEBYSHEV = np.inf

# decimal places in output
K_DECIMAL_PLACES = 3

# k parameter for the k-means clustering algorithms
K_PARAMETER = 5

# number of tests performed for each algorithm
K_NUMBER_OF_TESTS = 30

# depth of recursion in the binary search version of the algorithm
K_BINARY_SEARCH_DEPTH = 4

# enables and disables output of tests
K_TEST_OUTPUT_ENABLED = False

# enables the generation of distance tables
# if disabled, the table is read from a file, greatly reducing runtime
K_ENABLE_TABLE_GENERATION = False

# enables writing the table to an appropriate file
# if table generation is disabled, this also gets disabled
K_ENABLE_WRITING = True
