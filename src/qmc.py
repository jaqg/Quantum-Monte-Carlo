from debug import test_potential_ee, test_potential_eN, test_potential_NN
from hamiltonian import potential
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
    test_potential_ee()
    test_potential_eN()
    test_potential_NN()
    # print()

# -----------------------------------------------------------------------------
# Read Input
section('Hola')
# -----------------------------------------------------------------------------
r = (1., 2., 3., 4., 5., 6., 7., 8., 9.)
R = (0., 0., 0., 1., 1., 1.)
Z = [1, 2]
pot = potential(r, R, Z)
print('pot =', pot)
# print('pot =', pot, 'res =', 2./np.sqrt(27.) + 1./np.sqrt(108.))

# -----------------------------------------------------------------------------
end()
