import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
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
data = np.load("F444_tau_minus_20.npz")
flux_grid = data["fluxes_grid"]
masses = data["masses"] #same
redshift = data["redshift"]
masses = np.log10(masses)

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


# Confidence interval
redshift_idx = np.arange(len(redshift))
CI_flux = np.log10(np.vstack([compute_conf_ints(mstar_pdf[i], flux_grid[i,:]) for i in redshift_idx]))


fig, ax = plt.subplots(1, 1, figsize=(5,5))

low_z_colors = ['steelblue','lightskyblue','powderblue'] # ['brown','lightcoral','mistyrose']
colors = low_z_colors

ax.fill_between(z, CI_flux[:,0], CI_flux[:,6], alpha=1, color=colors[0])
ax.fill_between(z, CI_flux[:,1], CI_flux[:,5], alpha=1, color=colors[1])
ax.fill_between(z, CI_flux[:,2], CI_flux[:,4], alpha=1, color=colors[2])
ax.plot(z, CI_flux[:,3], linestyle='dotted', c='black')
# ax.plot(z, CI_baryon[:,6], linestyle='dashed', color='black')


#z of EazY
z_obs = np.array([10.47, 12.47, 12.10, 12.16, 9.92, 10.49, 11.19, 11.73, 11.88, 12.6, 12.7, 16.1])
zerr = np.array([[0.20, 0.02, 0.08, 0.27, 0.26, 0.22, 0.34, 0.18, 0.02, 2.5, 4.7, 1.6],
                [0.42, 0.27, 0.26, 0.23, 0.30, 0.28, 0.19, 0.23, 0.54, 0.1, 0.1, 2.4]])

# #F444W SE Flux values
# F = np.array([92.3, 47.3, 80.3, 142.5, 89.7, 58.4, 47.1, 44.5, 45.7, 27.1, 32.2, 21.1]) # Maybe need np.log F and F_err?
# log10F = np.log10(F)
# F_err = np.full((2, 12), 3.9)
# log10F_err = np.log10(F_err)

#F444W SE++ Flux values
# F = np.array([246, 226.2, 188.2, 364.2, 260.5,139.1, 132.9, 106.6, 106.4, 48.7, 41.3, 30.2]) # Maybe need np.log F and F_err?
# log10F = np.log10(F)
# F_err = np.array([8.1, 7.1, 7.3, 8.5, 7.7, 8.6, 6.9, 6.8, 5.1, 6.3, 5.8, 4.8])
# log10F_err = (1 / np.log(10)) * (F_err / F)

# #F150W SE flux values
# F = np.array([60.1, 12.1, 26.2, 18.7, 22.5, 37.1, 24.3, 20.6, 20.8, 0.0, 2.4, 3.4]) # Maybe need np.log F and F_err?
# log10F = np.log10(F)
# F_err = np.array([7.1, 6.3, 6.3, 6.4, 6.7, 6.5, 6.3, 6.8, 6.3, 7.3, 7.5, 6.3])
# log10F_err = np.array([np.log10(F_err) - F, F - np.log10(F_err)])


#F277W SE flux valuess
F = np.array([67.4, 56.2, 82.0, 94.2, 29.3, 46.3, 41.0, 43.5, 29.2, 44.6, 44.9, 27.8]) # Maybe need np.log F and F_err?
log10F = np.log10(F)
F_err = np.full((2, 12), 3.5)
log10F_err = np.log10(F_err)

# # #F277W SE++ flux valuess
# F = np.array([180.9, 248.2, 182.4, 235.8, 97.4, 110.7, 105.6, 103.7, 86.8, 82.9, 66.9, 50.9]) # Maybe need np.log F and F_err?
# log10F = np.log10(F)
# F_err = np.array([6.9, 6, 6, 6.3, 5.9, 6.9, 6, 5.8, 4.4, 5.9, 5.0, 4.5])
# log10F_err = (1 / np.log(10)) * (F_err / F)

# M_corr, _epsilon = eddington_bias(np.log10(10**M * (1./f_b)), M_err)
# M_corr = np.log10(10**M_corr * f_b) 



# plt.errorbar(z_obs, F, xerr=[zmin,zmax], yerr=Ferr, fmt='o', c='orange', label='Caputi+15')
ax.errorbar(z_obs, log10F, xerr=zerr, yerr=log10F_err, fmt='o', c='grey')
# ax.errorbar(z_obs, M_corr, xerr=zerr, yerr=M_err, fmt='o', c='orange', label='Casey24')

ax.set_xlim(7,18)
ax.set_ylim(-1,6)
ax.set_xlabel('$z$', size=17)
ax.set_ylabel("Log10(Flux) [nJy]", size = 17)
ax.text(0.05, 0.04, '$A = 1008 \; \mathrm{arcmin}^2$', size=12, color='black', alpha=0.8, transform = ax.transAxes)

leg = ax.legend(frameon=False, bbox_to_anchor=(0.44,0.19), fontsize=12, handletextpad=0.2) 
plt.gca().add_artist(leg) # Add the legend manually to the current Axes.

line1 = plt.Line2D((0,1),(0,0), color=colors[0], linewidth=5)
line2 = plt.Line2D((0,1),(0,0), color=colors[1], linewidth=5)
line3 = plt.Line2D((0,1),(0,0), color=colors[2], linewidth=5)
line4 = plt.Line2D((0,1),(0,0), color='black', linestyle='dotted', linewidth=2)
line5 = plt.Line2D((0,1),(0,0), color='black', linestyle='dashed', linewidth=2)
line_dummy = plt.Line2D((0,1),(0,0), color='white')
leg = ax.legend(handles=[line4,line5,line_dummy,line3,line2,line1], 
           labels=['$\mathrm{med}(M^{\star}_{\mathrm{max}})$','$f_{\star} = 1$; $+3\sigma$','',
               '$1\sigma$', '$2\sigma$', '$3\sigma$'],
                frameon=False, loc='upper right', fontsize=12, ncol=2)

vp = leg._legend_box._children[-1]._children[0] 
for c in vp._children: c._children.reverse() 
vp.align="right" 

info_text = (
    r"$\tau = -20$" + "\n"
    r"JWST/NIRCam Filter: F277W" + "\n"
    r"SE Model''"
)

anchored_text = AnchoredText(info_text, loc='lower left', frameon=True, prop=dict(size=11))
anchored_text.patch.set_alpha(0.8)
ax.add_artist(anchored_text)
plt.show()
plt.savefig('plots/evs_%s.pdf'%_obs_str, bbox_inches='tight', dpi=200)



