import os
import glob
import batman
import numpy as np
import pandas as pd
from tqdm import tqdm
import astropy.io.fits as fits
from astropy.io.fits.card import UNDEFINED


def generate_transit_params(n = 100000):
	M = np.linspace(0.3,2.0,n)
	N = 0.287971 * M**-2.35 * 0.001 * 20000
	N = N.astype(int)
	masses = [np.repeat(M[i],N[i]) for i in range(len(N))]
	mass_dist = np.concatenate(masses, axis=0)

	np.random.shuffle(mass_dist)
	mass_dist = mass_dist[:n]
	
	period = np.power(10, np.random.uniform(np.log10(2), np.log10(9), n))
	logRpRs = np.random.uniform(np.log10(20), np.log10(100), n) * -1 
	RpRs = np.power(10, logRpRs)
	t0 = np.random.uniform(0, 1, n)
	b = np.random.uniform(0, 1, n)
	a_vals = np.random.uniform(0.5, 0.7, n)
	aRs = 7.495**(1.0/3.0) *(10**-2) * 214.9394693836 * mass_dist**(4.0/3.0) * period**(2.0/3.0)# assume R ~ M 
	# print(mass_dist[0], period[0], (7.495*(10**-6) * 214.9394693836**3 * mass_dist[0]**3 * period[0]**2 * mass_dist[0]), aRs[0])
	return np.stack((t0,period,RpRs,aRs,b,a_vals, np.zeros_like(a_vals), np.zeros_like(a_vals), np.zeros_like(a_vals), np.zeros_like(a_vals)), axis=1)

def generate_transit_data(input_data, zeros, gen_params, TransitModel, params):
	params.t0 = gen_params[0] * gen_params[1]
	params.per = gen_params[1]  # orbital period
	params.rp = gen_params[2]  # planet radius (in units of stellar radii)
	params.a = gen_params[3]  # semi-major axis (in units of stellar radii)
	params.inc = 57.2957795 * np.arccos(gen_params[4]/gen_params[3])  # orbital inclination (in degrees)
	params.u = [gen_params[5]] # limb darkening coefficients [u1, u2, u3, u4]
	flux = TransitModel.light_curve(params)
	flux = np.log10(flux) * -2.5
	
	input_data = np.log10(input_data) * -2.5
	
	
	y = np.copy(input_data) * zeros
	mu_y = np.mean(y[y != 0])
	sigma_y = np.std(y[y != 0])
	y = (y - mu_y) / sigma_y
	y *= zeros
	
	x = np.copy(input_data)
	x = (flux + x) * zeros
	mu_x = np.mean(x[x != 0])
	sigma_x = np.std(x[x != 0])
	x = (x - mu_x) / sigma_x
	x *= zeros
	
	qn = np.sum(flux > 0)
	SNR = (qn**0.5) * (gen_params[2]**2 / sigma_x)
	np.reshape(x, (x.shape[0],1))
	np.reshape(y, (y.shape[0],1))
	np.reshape(flux, (flux.shape[0],1))
	return x, y, flux, SNR, mu_x, sigma_x, mu_y, sigma_y

def create_initial_batman(array_length=20610, total_time=28.625):
	params = batman.TransitParams()  # object to store transit parameters
	params.t0 = 0# time of inferior conjunction
	params.per = 1  # orbital period
	params.rp = 0.05  # planet radius (in units of stellar radii)
	params.a = 5  # semi-major axis (in units of stellar radii)
	params.inc = 90  # orbital inclination (in degrees)
	params.ecc = 0.0  # eccentricity
	params.w = 90.  # longitude of periastron (in degrees)
	params.limb_dark = "linear"  # limb darkening model
	params.u = [0.5] # limb darkening coefficients [u1, u2, u3, u4]

	t = np.linspace(0, total_time, array_length)  # times at which to calculate light curve
	m = batman.TransitModel(params, t)  # initializes model
	return m, params

def preprocess_data(header='../data/', out_dir='../data/processed/'):  
	if os.path.exists(out_dir): 
		print('exists!')
		return
	else:
		os.makedirs(out_dir, exist_ok=False)
	
	data_count = 9999 * 10
	file_count = 9999

	middle_count = data_count

	gen_parameters = generate_transit_params()
	transit_model, transit_params = create_initial_batman()

	file_id = -1
	file = None
	hdul = fits.open(header + 'sample_tess.fits')

	total_original =  np.zeros(dtype='float32', shape=(data_count, 1, 20610))
	total_features =  np.zeros(dtype='float32', shape=(data_count, 1, 20610))
	total_transits =  np.zeros(dtype='float32', shape=(data_count, 1, 20610))
	total_params =  np.zeros(dtype='float32', shape=(data_count, 10))


	files_mag = sorted(glob.glob(header + "mag_*.csv"))
	files = sorted(glob.glob(header + "sample_TESS_sim_mag_*.npy"))
	zeros = hdul[1].data['PDCSAP_FLUX']
	zeros[zeros != 0] = 1 # this array represents when we have tess data to simulate data gaps due to orbit
	for data in tqdm(range(middle_count)):
		if int(data / file_count) != file_id:  # since each file has 9999 stars we'll switch files each time we reach that numer
			file_id = int(data / file_count)
			file = np.load(files[file_id])
			mag_file = pd.read_csv(files_mag[file_id], header=None).values
		
		mag = mag_file[data % file_count,-1]
		if not isinstance(mag, float):
			mag = float(mag.replace('..','.'))
		x, y, transit, snr, mu_x, sigma_x, mu_y, sigma_y = generate_transit_data(file[data % file_count, :-1], zeros, gen_parameters[data], transit_model, transit_params)  # get the LCs
		gen_parameters[data] = [gen_parameters[data][0], np.log10(gen_parameters[data][1]),gen_parameters[data][2], gen_parameters[data][3], gen_parameters[data][4], gen_parameters[data][5], mu_x, sigma_x, mag, snr]

		total_features[data, 0, :] = x
		total_original[data, 0, :] = y
		total_transits[data, 0, :] = transit				
		total_params[data, :] = gen_parameters[data]		

	np.save(out_dir + 'total_added_t_sim.npy', total_features)
	np.save(out_dir + 'total_original_sim.npy', total_original)
	np.save(out_dir + 'total_transits_sim.npy', total_transits)
	np.save(out_dir + 'total_params_sim.npy', total_params)

	total_features_train = total_features[:10000]
	total_transits_train = total_transits[:10000]
	total_original_train = total_original[10000:20000]
	total_params_train = total_params[:10000]

	np.save(out_dir + 'total_added_t_sim_train.npy', total_features_train)
	np.save(out_dir + 'total_original_sim_train.npy', total_original_train)
	np.save(out_dir + 'total_transits_sim_train.npy', total_transits_train)
	np.save(out_dir + 'total_params_sim_train.npy', total_params_train)


	total_features_val = total_features[20000:30000]
	total_transits_val = total_transits[20000:30000]
	total_original_val = total_original[20000:30000]
	total_params_val = total_params[20000:30000]

	np.save(out_dir + 'total_added_t_sim_val.npy', total_features_val)
	np.save(out_dir + 'total_original_sim_val.npy', total_original_val)
	np.save(out_dir + 'total_transits_sim_val.npy', total_transits_val)
	np.save(out_dir + 'total_params_sim_val.npy', total_params_val)

	total_features_test = total_features[30000:]
	total_transits_test = total_transits[30000:]
	total_original_test = total_original[30000:]
	total_params_test = total_params[30000:]

	np.save(out_dir + 'total_added_t_sim_test.npy', total_features_test)
	np.save(out_dir + 'total_original_sim_test.npy', total_original_test)
	np.save(out_dir + 'total_transits_sim_test.npy', total_transits_test)
	np.save(out_dir + 'total_params_sim_test.npy', total_params_test)
