import time
import pandas as pd
import numpy as np


def read_points_and_distances(input_path='./input/list_of_points.csv', distances_path='./data/DistancesMatrix.xlsx'):
    points = None
    if 'csv' in input_path:
        points = pd.read_csv(input_path, header=None)
    elif 'xls' in input_path:
        points = pd.read_excel(input_path, header=None)

    distances = pd.read_excel(distances_path, header=0, index_col=0)
    distances.columns = range(distances.columns.size)
    distances.reset_index(drop=True, inplace=True)  # reset indexes

    return points, distances


def get_nodes(points, distances):
    if points is not None:
        nodes = points.to_numpy()[0]
        nodes = np.insert(nodes, 0, 0)
        # insert c in the list of nodes
        nodes = nodes.tolist()
    else:
        nodes = distances.columns.to_list()
    return nodes


def normalize_final_path(path):
    while path.index(0) != 0:
        path.append(path.pop(0))
    return path


def run_with_timer(func, *args):
    st = time.time()
    result = func(*args)
    duration = time.time() - st
    return result, duration

