import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

def fit_spline(time,flux,plot=False,nknots=None):
    # make sure data x vals are in increasing numbers
    order = np.argsort(time)
    flux = flux[order]
    time = time[order]
    
    # ensure no nans before fitting
    mask = ~np.isnan(flux) & ~np.isnan(time)
    flux = flux[mask]
    time = time[mask]

    if not nknots:
        # knot every 1.5 days...
        nknots = int((time[-1]-time[0])/0.1)
        print('default of 1 knot per 0.1 days...')
    knots = np.linspace(time[0],time[-1],nknots)
    
    # fit spline...need to check this is correct spline to be fitting
    tck = LSQUnivariateSpline(time,flux,t=knots[1:-1])
    
    if plot:
        y = tck(time)

        plt.scatter(time,flux,s=0.5)
        plt.title('Raw data')
        plt.show()

        plt.plot(time,y)
        plt.title('Spline fit')
        plt.show()
    return tck

class clean():
    def __init__(self, system_id, koi):
        data = Table.read('../fits/kplr{:09d}-2009166043257_llc.fits'.format(system_id))['TIME','PDCSAP_FLUX','PDCSAP_FLUX_ERR']
        self.time = np.array(data['TIME'])
        self.flux = np.array(data['PDCSAP_FLUX'])
        self.fluxerr = np.array(data['PDCSAP_FLUX_ERR'])
        self.period = koi.loc[system_id]['koi_period']
    
    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=(10,3))
        ax.plot(self.time, self.flux)
        
    def clean_arrays(self):
        
        print('no. datapoints prior to cleaning: {}'.format(len(self.time)))
        # make sure data x vals are in increasing numbers
        order = np.argsort(self.time)
        self.flux = self.flux[order]
        self.time = self.time[order]
        if isinstance(self.fluxerr,np.ndarray):
            self.fluxerr = self.fluxerr[order]

        # ensure no nans before fitting
        mask = ~np.isnan(self.flux) & ~np.isnan(self.time)
        self.flux = self.flux[mask]
        self.time = self.time[mask]
        print('no. datapoints after cleaning: {}'.format(len(self.time)))
        if isinstance(self.fluxerr,np.ndarray):
            self.fluxerr = self.fluxerr[mask]
            
    def fold_data_new(self, plot=True):
        self.time -= self.time[0]
        self.time = np.mod(self.time,self.period)
        if plot:
            plt.scatter(self.time,self.flux,s=0.5)
            plt.title('folded data on period %s days' %self.period)
            plt.show()
        
    def remove_stellar_activity(self,nknots=50):

        tck = fit_spline(self.time,self.flux,nknots=nknots)
        mask = np.ones(len(self.time),dtype=bool)
        converged = False
        n = 0
        while not converged:
            self.time = self.time[mask]
            self.flux = self.flux[mask]
            self.fluxerr = self.fluxerr[mask]
            tck = fit_spline(self.time,self.flux,nknots=nknots)
            y = tck(self.time)
            mask = (self.flux+3*self.fluxerr > y) & (self.flux-3*self.fluxerr < y)
            if mask.sum()==len(mask):
                converged=True
                print('spline-fit converged after %i iterations.' %n)
            else:
                n+=1

        plt.scatter(self.time,self.flux,s=0.5)
        plt.plot(self.time,tck(self.time))
        plt.title('data with spline overplotted')
        plt.show()

        self.flux -= tck(self.time)
        plt.scatter(self.time,self.flux,s=0.5)
        plt.title('data with spline removed')
        plt.show()