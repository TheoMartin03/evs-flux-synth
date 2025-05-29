import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.cosmology import Planck18 as cosmo
from unyt import Hz, Msun, Myr, erg, nJy, s, angstrom
from synthesizer import galaxy
from synthesizer.conversions import apparent_mag_to_fnu, fnu_to_lnu
from synthesizer.emission_models import PacmanEmission, IncidentEmission
from synthesizer.emission_models.attenuation import PowerLaw, Madau96
from synthesizer.instruments import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist


#Inputs filename for making repeated grids
filename = input("Enter a filename: ")


with h5py.File('/home/theo/dissertation/synthesizer/scripts/evstest2.h5', 'r') as hf:      # << change file directory here
    log10m = hf['log10m'][:]
    redshift = hf['z'][:]

# Define the grid
grid_name = "test_grid"
grid_dir = "../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)


model = IncidentEmission(grid=grid)

# Set up the SFH and metallicity distribution
sfh = SFH.Exponential(-20 * Myr, 1000 * Myr) ###vary tau values to generate varying grids
metal_dist = 0.01 

# Setting up filter from ["F115W", "F150W", "F444W", "F277W"]]
filter_codes = [f"JWST/NIRCam.{f}" for f in ["F277W"]] #Show predictions for different bands, F115W dropoff
fc = FilterCollection(filter_codes, new_lam=grid.lam)

# creating flux grid (supposedly quicker like this)
fluxes_grid = np.empty((len(redshift), len(log10m)))

# Loop over redshifts
for i, z in enumerate(redshift):
    print(f"Redshift z = {z}")
        # Creating Star object (exponential)
    exp_stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=10**log10m[0] * Msun,
        )

        # Creating galaxy object and getting flux
    gal = galaxy(stars=exp_stars, redshift=z)
    sed = gal.stars.get_spectra(model)
    sed.get_fnu(cosmo, z, Madau96)
    flux = sed.get_photo_fnu(fc)["JWST/NIRCam.F277W"]
# Loop over mass
    for j, mass in enumerate(log10m):

        fluxes_grid[i, j] = flux*(10**(mass-log10m[0]))


# Save results to file (have to manually move)
np.savez(filename,
         fluxes_grid=fluxes_grid,
         masses=log10m,
         redshift=redshift)


# # Plotting redshift logA relation
# plt.figure()
# plt.plot(redshift, logA_list, marker='o')
# plt.xlabel("Redshift")
# plt.ylabel("Log Flux Normalisation (logA)")
# plt.savefig("flux_normalisation_F277W", dpi=200, bbox_inches='tight')



# # Plotting flux and mass
# ax_lum.set_xlabel("Mass [M☉]")
# ax_lum.set_ylabel("Flux [nJy]")
# ax_lum.legend()
# plt.savefig("flux_mass_F444W.png", bbox_inches='tight', dpi=200)