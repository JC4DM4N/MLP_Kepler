"""
Quick sanity check for the data I've downloaded. Make sure
    KOI id matches those in KOI.csv
Expects data files will be saved at ../fits/.
"""

import pandas as pd
import numpy as np
import os
import re

files = os.listdir('../fits/.')

print('total of %i files' %len(files))

kepid = [re.search('kplr(.*)-', file).group(1).lstrip('0') for file in files]

sys = np.unique(kepid).astype(int)
nsys = len(sys)
print('total of %i systems' %nsys)

df = pd.read_csv('../KOI.csv')

present = np.in1d(sys,df.kepid)
print('%i out of %i systems found in KOI.csv' %(np.sum(present),nsys))
