import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from os import listdir

def fit_spline(time,flux,plot=False,nknots=None):
    """
    Fits a spline to the time, flux data

    Parameters
    ----------
    time : nd_array
    flux : nd_array
    plot : boolean
    	True to plot, False otherwise
    nkots: int
    	How tightly to fit the spline

    Returns
    -------
    tck : #james can you summarise

    """
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
    def __init__(self, system_id, koi, lc_index=None):
        """
        Initialise class clean()

        Parameters
        ----------
        system_id : int
        	kepler system id to be used
        koi : pandas.DataFrame
        	DataFrame of the kepler koi meta data
        lc_index: None or int
        	if None, load and concatenate all data for the system
        	if integer n, load the n'th quarter for the system (starts from zero)

        """
        self.period = koi.loc[system_id]['koi_period']
        #finds the date part of the filename
        fnames = sorted(listdir('../fits/slc/'))
        fnames = [fname[14:31] for fname in fnames if (fname[4:13] == '{:09d}'.format(system_id))]
        if lc_index == None:
            t = np.empty(0)
            f = np.empty(0)
            e = np.empty(0)
            for fname in fnames:
                data = Table.read('../fits/slc/kplr{:09d}-{}.fits'.format(system_id, fname))['TIME','PDCSAP_FLUX','PDCSAP_FLUX_ERR']
                t = np.append(t, data['TIME'])
                f = np.append(f, data['PDCSAP_FLUX'] - np.nanmean(data['PDCSAP_FLUX']) + 1e4) # subtract the mean
                e = np.append(e, data['PDCSAP_FLUX_ERR'])
            else:
            	data = Table.read('../fits/slc/kplr{:09d}-{}.fits'.format(system_id, fnames[lc_index]))['TIME','PDCSAP_FLUX','PDCSAP_FLUX_ERR']
            	t = np.array(data['TIME'])
            	f = np.array(data['PDCSAP_FLUX'])
            	e = np.array(data['PDCSAP_FLUX_ERR'])

        self.time = t
        self.flux = f
        self.fluxerr = e
        self.n = len(t)

    def plot(self):
        """
        Simple scatter plot

        Parameters
        ----------
        param : type
        	desc

        Returns
        -------
        value : type
        	desc
        """
        fig, ax = plt.subplots(1,1, figsize=(20,5))
        ax.scatter(self.time, self.flux, s=0.1)
        ax.set(ylim = [self.flux.mean()-self.flux.std()*6, self.flux.mean()+self.flux.std()*6])

    def clean_arrays(self, verbose=False):

        if verbose:
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
        if verbose:
            print('no. datapoints after cleaning: {}'.format(len(self.time)))
        if isinstance(self.fluxerr,np.ndarray):
            self.fluxerr = self.fluxerr[mask]

    def fold_data(self, plot=True):
        self.time -= self.time[0]
        self.time = np.mod(self.time,self.period)
        if plot:
            plt.scatter(self.time,self.flux,s=0.5)
            plt.title('folded data on period %s days' %self.period)
            plt.figure(figsize=(20,5))
            plt.show()

    def RMSE(y1, y2):
        return np.sqrt(np.mean((y1-y2)**2))

    def cross_validation(time,flux):
        """
        fit spline to half of the data, then minimise RMSE
            when using the other half of the data.

        bounds: np.array([min, max])
            upper and lower bounds of knots range
        returns:
            nopt: optimal number of knots
            errmin: RMSE of optimal fit
        """
        errors = []
        # may or may not want this random seed...
        np.random.seed(111)

        # initial guess a knot every 1.5 days
        n0 = (time[-1]-time[0])/1.5
        # define bounds as +/- 25 knots either side of n0
        bounds = np.clip(np.linspace(n0-25,n0+25,50).astype(int),1,500)
        for nknots in bounds:
            # randomly mask half the data before fitting
            mask = time<np.mean(time)
            np.random.shuffle(mask)
            #force mask[0] and mask[-1] to True
            mask[0] = True
            mask[-1] = True
            # fit spline to masked data
            try:
                spl = fit_spline(time[mask],flux[mask],nknots=nknots)
                # predict for other half of data and calc errors
                errors.append(RMSE(flux[~mask],spl(time[~mask])))
            except:
                # if fit fails for whatever reason append error=inf
                # usually this is because number of knots exceeds the max allowed
                errors.append(np.inf)
        errors = np.asarray(errors)

        # optimal number of knots
        nopt = bounds[errors==np.min(errors)][0]
        return nopt, np.min(errors)

    def remove_stellar_activity(obs, verbose=False, plot=False):

        time = obs[1].data.TIME
        flux = obs[1].data.PDCSAP_FLUX
        fluxerr = obs[1].data.PDCSAP_FLUX_ERR

        time,flux,fluxerr = clean_arrays(time,flux,fluxerr)

        # get optimal number of knots using cross validation
        nopt, errmin = cross_validation(time,flux)
        if verbose:
            print('optimal number of knots found: %i, err: %.3f' %(nopt, errmin))
        spl = fit_spline(time, flux, nknots=nopt)

        flux -= spl(time)
        fluxerr -= spl(time)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
            ax1.scatter(time,flux+spl(time),s=0.5)
            ax1.plot(time,spl(time),c='red')
            ax1.set_title('data with spline overplotted')

            ax2.scatter(time,flux,s=0.5)
            ax2.set_title('data with spline removed')
            plt.show()

        return time,flux,fluxerr
