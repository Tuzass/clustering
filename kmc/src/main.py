from functions import *
import numpy as np

statlog_vehicle_file = open(r'uci-datasets/yeast/data.csv', mode='r')
data, k = readUCI(statlog_vehicle_file)

print (f'k = {k}\nnumber of instances = {len(data)}')
