"""
Plot flux against time for a chosen fits file
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file')
args = parser.parse_args()
file = args.file

obs = fits.open(file)

times = obs[1].data['TIME']
flux = obs[1].data['PDCSAP_FLUX']

plt.scatter(times,flux,s=0.5)
plt.show()
