from debug import all_debug
from hamiltonian import phi, d_phi, d2_phi, psi, d_psi, d2_psi, potential_ee, potential_eN, potential_NN, potential, kinetic_e, kinetic_N, kinetic, e_loc
from IO import title, section, end

import numpy as np

# =============================================================================
# Start of the program
# =============================================================================
title('Quantum Monte Carlo')
section('Initialization')

# -----------------------------------------------------------------------------
# Debugging
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print('Debugging:')
    all_debug()

# -----------------------------------------------------------------------------
# Read Input
section('Pruebas')
# -----------------------------------------------------------------------------
a = 1.2
r_SI = (1., 2., 3., 4., 5., 6., 7., 8., 9.)
R_SI = (0., 0., 0., 1., 1., 1.)
Z = (1, 2)

angstrom_to_au = 1.8897259886
r = []
for i in r_SI:
    r.append(i * angstrom_to_au)

R = []
for i in R_SI:
    R.append(i * angstrom_to_au)

print('a =', a)
print('Z =', Z)
print('r (Angstrom) =', r_SI)
print('R (Angstrom) =', R_SI)
print('r (a.u.) =', r)
print('R (a.u.) =', R)
print('\n')

vee = potential_ee(r)
veN = potential_eN(r, R, Z)
vNN = potential_NN(R, Z)
vtot = potential(r, R, Z)
print('Vee =', vee)
print('VeN =', veN)
print('VNN =', vNN)
print('Vtot =', vtot)
print('\n')

phi_r = phi(a, r, R)
dphi = d_phi('x', a, r, R)
print('phi =', phi_r, ', dphi =', dphi, ', d2phi(x) =', d2_phi('x', a, r, R))

psi_r = psi(a, r, R)
dpsi = d_psi('x', a, r, R)
print('psi =', psi_r, ', dpsi =', dpsi, ', d2psi(x) =', d2_psi('x', a, r, R))
print('\n')

ke = kinetic_e(a, r, R)
kN = kinetic_N()
ktot = kinetic(a, r, R)
print('Ke =', ke)
print('KN =', kN)
print('Ktot =', ktot)
print('\n')

eloc = e_loc(a, r, R, Z)
print('E_loc = Ktot + Vtot = ', eloc)
print('\n')

# -----------------------------------------------------------------------------
end()
