import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from matplotlib.lines import Line2D
from unyt import Hz, Msun, Myr, erg, nJy, s, angstrom

from synthesizer import galaxy
from synthesizer.conversions import apparent_mag_to_fnu, fnu_to_lnu
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.instruments import Filter, FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist

# Set up a figure to plot on
fig = plt.figure()
ax_lum = fig.add_subplot(111)

# Logscale
ax_lum.loglog()


# Define the grid
grid_name = "test_grid"
grid_dir = "../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the emission model
model = PacmanEmission(
    grid,
    fesc=0.5,
    fesc_ly_alpha=0.5,
    tau_v=0.1,
    dust_curve=PowerLaw(slope=-1),
)

# Decide Redshift
redshift = 12

# Set up the SFH
sfh = SFH.Exponential(20 * Myr, 1000 * Myr)

# Set up the metallicity distribution (may be larger for massive galaxies?)
metal_dist = ZDist.Normal(mean=0.01, sigma=0.005)

# Set up masses to loop over + colours:
masses = np.logspace(8,12,num=10)
colour_array = plt.cm.plasma(np.linspace(0,1,len(masses)))

# COSMOS-Web filters
filter_codes = [f"JWST/NIRCam.{f}" for f in ["F115W", "F150W", "F444W", "F277W"]]
fc = FilterCollection(filter_codes, new_lam=grid.lam)

# Creating specific filter - maybe not needed
f115w = Filter("JWST/NIRCam.F115W", new_lam=grid.lam)
f115w_min = Filter.min(f115w)
f115w_max = Filter.max(f115w)

# Looping
for i, mass in enumerate(masses):

    # Get the stellar population
    exp_stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass= mass * Msun,
    )

    # Creating galaxy object
    gal = galaxy(stars=exp_stars, redshift=redshift)

    # Generate spectra of galaxy
    gal_spectra = gal.stars.get_spectra(model)

    # Calculating luminosity of galaxy
    lums = gal_spectra.get_photo_lnu(fc)
    lums.plot_photometry(show=True)
    
    # Plotting luminosity
    ax_lum.plot(
    gal.stars.spectra["attenuated"]._lam,
    gal.stars.spectra["attenuated"]._lnu,
    color= colour_array[i],
    linestyle= "-",
    label=f"Mass = {mass:.1e} Msun"
    )

# Labels + limits
ax_lum.set_xlim(10, 1e13)  # Set x-axis range (for wavelength in Angstrom)
ax_lum.set_ylim(1e5, 1e35)  # Set y-axis range (for luminosity in erg/s/Hz)
ax_lum.set_xlabel("Wavelength [Å]")
ax_lum.set_ylabel("Luminosity [erg/s/Hz]")
ax_lum.legend(title="Masses", loc="lower right")
plt.savefig("luminosity_Wavelength_filt", bbox_inches='tight', dpi=200)  