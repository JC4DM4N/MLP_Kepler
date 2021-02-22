import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

def clean_arrays(time,flux,fluxerr=None):
    print('no. datapoints prior to cleaning: '.format(len(time)))
    # make sure data x vals are in increasing numbers
    order = np.argsort(time)
    flux = flux[order]
    time = time[order]
    if isinstance(fluxerr,np.ndarray):
        fluxerr = fluxerr[order]

    # ensure no nans before fitting
    mask = ~np.isnan(flux) & ~np.isnan(time)
    flux = flux[mask]
    time = time[mask]
    print('no. datapoints after cleaning: '.format(len(time)))
    if isinstance(fluxerr,np.ndarray):
        fluxerr = fluxerr[mask]
        return time, flux, fluxerr
    else: 
        return time, flux

def fold_data(obs,sysid,df,planetid=None):
    """
    fold the data back on itself given the planet's orbital period.
    """
    times = obs[1].data.TIME
    flux = obs[1].data.PDCSAP_FLUX
    if planetid:
        period = df[df.kepid==sysid]['koi_period'].iloc[planetid]
    else:
        period = df[df.kepid==sysid]['koi_period'].iloc[0]
    
    times -= times[0]
    times = np.mod(times,period)
    
    plt.scatter(times,flux,s=0.5)
    #plt.ylim([8230,8290])

    return times, flux

def fold_data_new(time,flux,period,plot=True):
    time -= time[0]
    time = np.mod(time,period)
    if plot:
        plt.scatter(time,flux,s=0.5)
        plt.title('folded data on period %s days' %period)
        plt.show()
    return time,flux

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

def remove_stellar_activity(obs,nknots=50):
    
    time = obs[1].data.TIME
    flux = obs[1].data.PDCSAP_FLUX
    fluxerr = obs[1].data.PDCSAP_FLUX_ERR
    
    time,flux,fluxerr = clean_arrays(time,flux,fluxerr)
    
    tck = fit_spline(time,flux,nknots=nknots)
    mask = np.ones(len(time),dtype=bool)
    converged = False
    n = 0
    while not converged:
        time = time[mask]
        flux = flux[mask]
        fluxerr = fluxerr[mask]
        tck = fit_spline(time,flux,nknots=nknots)
        y = tck(time)
        mask = (flux+3*fluxerr > y) & (flux-3*fluxerr < y)
        if mask.sum()==len(mask):
            converged=True
            print('spline-fit converged after %i iterations.' %n)
        else:
            n+=1

    # fit converged spline back on original data...
        
    time = obs[1].data.TIME
    flux = obs[1].data.PDCSAP_FLUX
    fluxerr = obs[1].data.PDCSAP_FLUX_ERR
    
    time,flux,fluxerr = clean_arrays(time,flux,fluxerr)
    
    plt.scatter(time,flux,s=0.5)
    plt.plot(time,tck(time))
    plt.title('data with spline overplotted')
    plt.show()

    flux -= tck(time)
    plt.scatter(time,flux,s=0.5)
    plt.title('data with spline removed')
    plt.show()
    
    return time,flux,fluxerr,tck