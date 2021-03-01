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

	def fold_data(self, plot=True):
		self.time -= self.time[0]
		self.time = np.mod(self.time,self.period)
		if plot:
			plt.scatter(self.time,self.flux,s=0.5)
			plt.title('folded data on period %s days' %self.period)
			plt.figure(figsize=(20,5))
			plt.show()

	def remove_stellar_activity(self,nknots=50):
		# TODO - make copies of self.time and self.flux because they are currently being overwritten when running this.
		# would be better if this function returned the cleaned flux then we can keep running this function with different
		# nknots until we reach a good fit.
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
		
		mask = abs(self.time-self.time[self.n//2])<10
		
		fig, ax = plt.subplots(4,1, figsize=(20,15))
		
		ax[0].scatter(self.time,self.flux,s=0.5)
		ax[0].plot(self.time,tck(self.time))
		ax[0].title.set_text('data with spline overplotted')
		
		ax[1].scatter(self.time[mask],self.flux[mask],s=0.5)
		ax[1].plot(self.time[mask],tck(self.time[mask]))
		ax[1].title.set_text('data with spline overplotted, 10 day window')

		self.flux -= tck(self.time)
		ax[2].scatter(self.time,self.flux,s=0.5)
		ax[2].title.set_text('data with spline removed')

		ax[3].scatter(self.time[mask],self.flux[mask],s=0.5)
		ax[3].title.set_text('data with spline removed, 10 day window')