import matplotlib.pyplot as plt
import pandas as pd

from IO import read_xyz, read_input, print_table, section, title, end
from hydrogen import e_loc
from MonteCarlo import estimated_energy, variance_eloc, VMC, PDMC, output_VMC
from debug import test_potential
from plots import plot_eloc_a, plot_eloc_xcoord, plot_psi_xcoord

style_file = 'mine.mplstyle'
plt.style.use(style_file)

colores=['#56B4E9', '#E69F00', '#009E73', '#0072B2', '#D55E00', '#CC79A7', '#F0E442']

# =============================================================================
# Start of the program
title('Quantum Monte Carlo')
section('Initialization')
# =============================================================================

# -----------------------------------------------------------------------------
# Debugging
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print('Debugging:')
    test_potential()
    print()

# -----------------------------------------------------------------------------
# Read Input
# -----------------------------------------------------------------------------
print('Reading input...')
calculation_type, a, nmc, mc_trials, lim, dt, tau, eref = read_input('INPUT')
nxyz = read_xyz('XYZ')

# =============================================================================
# Find eigenfunctions
# =============================================================================
# The eigenfunctions of H correspond to the eigenfunctions that make E_L 
# constant (for the correct values of a, r)
#
# The energy reads:
# 
# E = - 1/2 a**2 + (a - 1)/|r|
#
# Therefore, the energy is constant and equal to -0.5 Ha for a = 1.0

# -----------------------------------------------------------------------------
# Find the value of a
section('Numerical estimation of the energy')
# -----------------------------------------------------------------------------
# Define a position
r = (0.1, 0.2, 0.3)

# The optimal value is
a_opt = 1.
local_energy = e_loc(a_opt, r)

# -----------------------------------------------------------------------------
# Plot the local energy along the x axis
# -----------------------------------------------------------------------------

# Store the desired values of a in a list
a_list = [0.1, 0.2, 0.5, 1., 1.5, 2.]
print('Saving plots in ./plots/')
plot_eloc_a(r, 2.0, 'plots/local-energy-vs-a.pdf')
plot_eloc_xcoord(a_list, 4, 'plots/local-energy.pdf')
plot_psi_xcoord([a_opt], 4, 'plots/E-Psi.pdf')

# -----------------------------------------------------------------------------
# Numerical estimation of the energy
# -----------------------------------------------------------------------------
print('Computing energy...\n')
ninterval = 50
num_lim = 5.
estimated_E = estimated_energy(ninterval, num_lim, a_list)

# Compute the variance of the local energy
variance_E = variance_eloc(ninterval, num_lim, a_list)

data = pd.DataFrame({'a': a_list, 'E': estimated_E, 'sigma^2': variance_E})
print_table("Estimated energies:", data)

# =============================================================================
# Quantum Monte Carlo
# =============================================================================

if calculation_type == 'VMC':
    # -------------------------------------------------------------------------
    # Variational Monte Carlo
    # -------------------------------------------------------------------------
    # TODO: pasar r como input
    sVMC, syMe, geMe = VMC(nxyz['nelectrons'], a, mc_trials, nmc, lim, dt)
    output_VMC(sVMC, syMe, geMe)

elif calculation_type == 'PDMC':
    # -------------------------------------------------------------------------
    # Pure diffusion Monte Carlo
    # -------------------------------------------------------------------------
    # TODO: loop sobre cada atomo y para nelectrons
    # TODO: pasar r como input
    PDMC(a, mc_trials, nmc, dt, tau, eref)

else:
    print("Error: unknown calculation type")
    end()
    exit(1)

# -----------------------------------------------------------------------------
end()
