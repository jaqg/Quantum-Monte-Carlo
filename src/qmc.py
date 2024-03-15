from IO import title, section, read_input, read_xyz, extract_R_ne, print_input, print_MC_data, end
from debug import all_debug
from hamiltonian import phi, d_phi, d2_phi, psi, d_psi, d2_psi, potential_ee, potential_eN, potential_NN, potential, kinetic_e, kinetic_N, kinetic, e_loc
from MonteCarlo import average, error, MC, Metropolis_symmetric_MC, Metropolis_generalized_MC, Pure_diffusion_MC, VMC, output_VMC, PDMC, output_PDMC

import numpy as np

# =============================================================================
# Start of the program
# =============================================================================
title('Quantum Monte Carlo')

# -----------------------------------------------------------------------------
# Debugging
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    section('Debugging')
    all_debug()

# -----------------------------------------------------------------------------
# Read Input
section('Input')
# -----------------------------------------------------------------------------
print('Reading input...\n')
calculation_type, a, nmc, mc_trials, dt_VMC, dt_metro, dt_PDMC, tau, eref = read_input('INPUT')
title, charge, nxyz = read_xyz('XYZ')
ne, R, Z = extract_R_ne(charge, nxyz)

# -----------------------------------------------------------------------------
# Output file
# -----------------------------------------------------------------------------
output_filename = 'OUTPUT'

# -----------------------------------------------------------------------------
# Print Input
# -----------------------------------------------------------------------------
print_input(title, a, charge, nxyz, ne, R, Z)

#-----------------------------------------------------------------------------
section('MC')
#-----------------------------------------------------------------------------
# Print MC input data
print_MC_data(mc_trials, nmc, dt_VMC, dt_metro, dt_PDMC, tau, eref)

# Perform the actual calculation
if calculation_type == 'VMC':
    # -------------------------------------------------------------------------
    # Variational Monte Carlo
    # -------------------------------------------------------------------------
    sVMC, syMe, geMe = VMC(a, ne, R, Z, mc_trials, nmc, dt_VMC, dt_metro)
    output_VMC(sVMC, syMe, geMe)

elif calculation_type == 'PDMC':
    # -------------------------------------------------------------------------
    # Pure diffusion Monte Carlo
    # -------------------------------------------------------------------------
    sPDMC = PDMC(a, ne, R, Z, mc_trials, nmc, dt_PDMC, tau, eref)
    output_PDMC(sPDMC)

else:
    print("Error: unknown calculation type")
    end()
    exit(1)

# -----------------------------------------------------------------------------
end()
