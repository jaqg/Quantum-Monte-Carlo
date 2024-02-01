from IO import print_table, section, title, end
from hydrogen import e_loc, psi, estimated_energy, variance_eloc, average, variance, error, MonteCarlo, Metropolis_symmetric_MC, Metropolis_generalized_MC, Pure_diffusion_MC
from debug import test_potential
from plots import plot_eloc_a, plot_eloc_xcoord, plot_psi_xcoord
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
from prettytable import PrettyTable
import pandas as pd

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
a = 1.
local_energy = e_loc(a, r)

# -----------------------------------------------------------------------------
# Plot the local energy along the x axis
# -----------------------------------------------------------------------------

# Store the desired values of a in a list
a_list = [0.1, 0.2, 0.5, 1., 1.5, 2.]
print('Saving plots in ./plots/')
plot_eloc_a(r, 2.0, 'plots/local-energy-vs-a.pdf')
plot_eloc_xcoord(a_list, 4, 'plots/local-energy.pdf')
plot_psi_xcoord([a], 4, 'plots/E-Psi.pdf')

# -----------------------------------------------------------------------------
# Numerical estimation of the energy
# -----------------------------------------------------------------------------
print('Computing energy...\n')
ninterval = 50
lim = 5.
estimated_E = estimated_energy(ninterval, lim, a_list)

# Compute the variance of the local energy
variance_E = variance_eloc(ninterval, lim, a_list)

data = pd.DataFrame({'a': a_list, 'E': estimated_E, 'sigma^2': variance_E})
print_table("Estimated energies:", data)

# =============================================================================
# Variational Monte Carlo
# =============================================================================

# -----------------------------------------------------------------------------
# Monte Carlo algorithm
section('Variational Monte Carlo')
# -----------------------------------------------------------------------------

# Perform 30 MC simulations with NMC steps
mc_trials = 30
a = 1.2
nmc = 1000
lim = 5.
mc_energy = []
for i in range(mc_trials):
    mc_energy.append(MonteCarlo(a, nmc, lim))

# Print results
variational_mc_E = average(mc_energy)
variational_mc_sigma = error(mc_energy)
print('E = ', variational_mc_E, '+-', variational_mc_sigma)

# -----------------------------------------------------------------------------
# Metropolis algorithm
section('Metropolis (symmetric) MC')
# -----------------------------------------------------------------------------
dt = 1.

metropolis_sym_E = []
metropolis_sym_ratio = []
for i in range(mc_trials):
    x, y = Metropolis_symmetric_MC(a, nmc, dt)
    metropolis_sym_E.append(x)
    metropolis_sym_ratio.append(y)

# Print results
sym_metropolis_E = average(metropolis_sym_E)
sym_metropolis_sigma_E = error(metropolis_sym_E)
sym_metropolis_ratio = average(metropolis_sym_ratio)
sym_metropolis_sigma_ratio = error(metropolis_sym_ratio)

print('E = ', sym_metropolis_E, '+-', sym_metropolis_sigma_E)
print('Ratio = ', sym_metropolis_ratio, '+-', sym_metropolis_sigma_ratio)

section('Metropolis (generalized) MC')

metropolis_gen_E = []
metropolis_gen_ratio = []
for i in range(mc_trials):
    x, y = Metropolis_generalized_MC(a, nmc, dt)
    metropolis_gen_E.append(x)
    metropolis_gen_ratio.append(y)

# Print results
gen_metropolis_E = average(metropolis_gen_E)
gen_metropolis_sigma_E = error(metropolis_gen_E)
gen_metropolis_ratio = average(metropolis_gen_ratio)
gen_metropolis_sigma_ratio = error(metropolis_gen_ratio)

print('E = ', gen_metropolis_E, '+-', gen_metropolis_sigma_E)
print('Ratio = ', gen_metropolis_ratio, '+-', gen_metropolis_sigma_ratio)

# -----------------------------------------------------------------------------
# Pure diffusion Monte Carlo
section('Pure Diffusion MC')
# -----------------------------------------------------------------------------
dt = 0.05
tau = 100.
eref = -0.5

pure_diffusion_E = []
pure_diffusion_ratio = []
for i in range(mc_trials):
    x, y = Pure_diffusion_MC(a, nmc, dt, tau, eref)
    pure_diffusion_E.append(x)
    pure_diffusion_ratio.append(y)

# Print results
pd_E = average(pure_diffusion_E)
pd_sigma_E = error(pure_diffusion_E)
pd_ratio = average(pure_diffusion_ratio)
pd_sigma_ratio = error(pure_diffusion_ratio)

print('E = ', pd_E, '+-', pd_sigma_E)
print('Ratio = ', pd_ratio, '+-', pd_sigma_ratio)

# =============================================================================
# Summary
section('Summary')
# =============================================================================
methods = ['VMC', 'Symmetric Metropolis', 'Generalized Metropolis', 'PDMC']

energies = [variational_mc_E,
            sym_metropolis_E,
            gen_metropolis_E,
            pd_E]

errors = [variational_mc_sigma,
          sym_metropolis_sigma_E,
          gen_metropolis_sigma_E,
          pd_sigma_E]

ratios = [0.,
          sym_metropolis_ratio,
          gen_metropolis_ratio,
          pd_ratio]

sigma_ratios = [0.,
                sym_metropolis_sigma_ratio,
                gen_metropolis_sigma_ratio,
                pd_sigma_ratio]

data = pd.DataFrame({'Method': methods, 'Energy, E': energies, 'Sigma_E': errors,
                     'Ratio, A': ratios, 'Sigma_A': sigma_ratios})
print_table("", data)

# -----------------------------------------------------------------------------
end()
