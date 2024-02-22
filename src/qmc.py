from debug import test_potential_ee, test_potential_eN
from hamiltonian import potential_ee, potential_eN
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
    # print()

# -----------------------------------------------------------------------------
# Read Input
section('Hola')
# -----------------------------------------------------------------------------
# r = (1., 2., 3., 4., 5., 6., 7., 8., 9.)
r = (1., 2., 3.)
R = (0., 0., 0.)
Z = [1]
pot = potential_eN(r, R, Z)
print('pot =', pot)
# print('pot =', pot, 'res =', 2./np.sqrt(27.) + 1./np.sqrt(108.))

# -----------------------------------------------------------------------------
end()
