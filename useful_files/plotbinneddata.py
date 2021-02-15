from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.stats import binned_statistic

parser = argparse.ArgumentParser()
parser.add_argument('-file')
parser.add_argument('-period')
parser.add_argument('-duration')
args = parser.parse_args()
file = args.file
period = 3.709870645 #days #args.period
duration = 3.1/24. #days

dat = fits.open(file)

times = []
flux = []
fluxerr = []
for i,_ in enumerate(dat[1].data):
    times.append(dat[1].data[i][0])
    flux.append(dat[1].data[i][7])
    fluxerr.append(dat[1].data[i][8])

data = np.asarray([times,flux,fluxerr]).T
#remove nan values
data = data[~np.isnan(data[:,0])]
data = data[~np.isnan(data[:,1])]
data = data[~np.isnan(data[:,2])]

plt.figure('original')
plt.scatter(data[:,0],data[:,1],s=0.5)

#bin data
binwidth = duration/3.
nbins = int((data[-1,0]-data[0,0])*20)
binned = binned_statistic(data[:,0],data[:,1],statistic='mean',bins=nbins)

plt.figure('binned')
plt.scatter(binned[1][:-1],binned[0], s=1)
plt.show()

import pdb; pdb.set_trace()
