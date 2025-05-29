import numpy as np
import h5py
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
import astropy.units as u
from scipy import integrate
from evstats import evs
from evstats.stats import compute_conf_ints, eddington_bias
from evstats.stellar import apply_fs_distribution


with h5py.File('evstest2','r') as hf:
    log10m = hf['log10m'][:]
    f = hf['f'][:]
    F = hf['F'][:]
    N = hf['N'][:]
    z = hf['z'][:]


# Loading Flux grid data
datam50 = np.load("tau_minus_50.npz")
flux_gridm50 = datam50["fluxes_grid"]
redshiftm50 = datam50["redshift"]

# Loading Flux grid data
datam40 = np.load("tau_minus_40.npz")
flux_gridm40 = datam40["fluxes_grid"]
redshiftm40 = datam40["redshift"]

# Loading Flux grid data
datam30 = np.load("tau_minus_30.npz")
flux_gridm30 = datam30["fluxes_grid"]
redshiftm30 = datam30["redshift"]

# Loading Flux grid data
datam20 = np.load("tau_minus_20.npz")
flux_gridm20 = datam20["fluxes_grid"]
redshiftm20 = datam20["redshift"]

# Loading Flux grid data
datam10 = np.load("tau_minus_10.npz")
flux_gridm10 = datam10["fluxes_grid"]
redshiftm10 = datam10["redshift"]



_obs_str = 'Casey24_EazY_FLUX'

# Cosmology Values etc.
whole_sky = (41252.96 * u.deg**2).to(u.arcmin**2)
survey_area = 1008 * u.arcmin**2
fsky = float(survey_area / whole_sky)
phi_max = evs._apply_fsky(N, f, F, fsky)
f_b = 0.156

CI_mhalo = compute_conf_ints(phi_max, log10m)
# CI_baryon = np.log10(10**CI_mhalo * f_b) # Probably need to change this too? 


mstar_pdf = np.vstack([apply_fs_distribution(_phi_max, log10m, _N=int(1e4), method='lognormal')   #change N for load time/ smoothness, 3 = quick, 5 = precise
        for _phi_max in phi_max])


#Computing fluxes for each different tau value
redshift_idxm10 = np.arange(len(redshiftm10))
CI_fluxm10 = np.log10(np.vstack([compute_conf_ints(mstar_pdf[i], flux_gridm10[i,:]) for i in redshift_idxm10]))

redshift_idxm20 = np.arange(len(redshiftm20))
CI_fluxm20 = np.log10(np.vstack([compute_conf_ints(mstar_pdf[i], flux_gridm20[i,:]) for i in redshift_idxm20]))

redshift_idxm30 = np.arange(len(redshiftm30))
CI_fluxm30 = np.log10(np.vstack([compute_conf_ints(mstar_pdf[i], flux_gridm30[i,:]) for i in redshift_idxm30]))

redshift_idxm40 = np.arange(len(redshiftm40))
CI_fluxm40 = np.log10(np.vstack([compute_conf_ints(mstar_pdf[i], flux_gridm40[i,:]) for i in redshift_idxm40]))

redshift_idxm50 = np.arange(len(redshiftm50))
CI_fluxm50 = np.log10(np.vstack([compute_conf_ints(mstar_pdf[i], flux_gridm50[i,:]) for i in redshift_idxm50]))




fig, ax = plt.subplots(1, 1, figsize=(5,5))

low_z_colors = ['steelblue','lightskyblue','powderblue'] # ['brown','lightcoral','mistyrose']
colors = low_z_colors


ax.plot(redshiftm50, CI_fluxm50[:, 3], color='tab:purple', label=r'$\tau = -50$')
ax.plot(redshiftm40, CI_fluxm40[:, 3], color='tab:brown', label=r'$\tau = -40$')
ax.plot(redshiftm30, CI_fluxm30[:, 3], color='tab:gray', label=r'$\tau = -30$')
ax.plot(redshiftm20, CI_fluxm20[:, 3], color='tab:blue', label=r'$\tau = -20$')
ax.plot(redshiftm10, CI_fluxm10[:, 3], color='tab:orange', label=r'$\tau = -10$')


ax.set_xlim(8,16)
ax.set_ylim(1,5)
ax.set_xlabel('$z$', size=17)
ax.set_ylabel("Log10(Flux) [nJy]", size = 17)
ax.text(0.05, 0.04, '$A = 1008 \; \mathrm{arcmin}^2$', size=12, color='black', alpha=0.8, transform = ax.transAxes)

leg = ax.legend(frameon=False, bbox_to_anchor=(0.44,0.19), fontsize=12, handletextpad=0.2) 
plt.gca().add_artist(leg) # Add the legend manually to the current Axes.

ax.legend(frameon=False, fontsize=12, loc='upper right')


plt.show()
plt.savefig('plots/evs_%s.pdf'%_obs_str, bbox_inches='tight', dpi=200)