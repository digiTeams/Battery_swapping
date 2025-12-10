'''
Parallel computation of bundle costs
'''


import copy
from datetime import datetime, timedelta
import math
import numpy as np
from collections import OrderedDict
from geopy.distance import geodesic, great_circle
from multiprocessing import Pool

from config import data_params, env_params

class ParallelTasks:
    def __init__(self, agents, requests, bundles, BS_pairs, charges, demands):
        self.agents = agents
        self.requests = requests
        self.bundles = bundles
        self.BS_pairs = BS_pairs
        self.charges = charges
        self.demands = demands
        self.results = {}


    #Single task computation
    def eval(self):

        return






    # Parallel task computation
    def pool_compute(self):


        return
