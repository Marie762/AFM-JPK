# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""

import os
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas
import afmformats

import extractJPK

# extract the data from all the jpk-force files in the directory 'Data'
d, F, t = extractJPK.force()

print(d)
print(F)
print(t)