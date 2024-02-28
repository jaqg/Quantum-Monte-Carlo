from hamiltonian import potential_ee, potential_eN, potential_NN

import numpy as np

def test_potential_ee():
    expected_output = 1./np.sqrt(14.)
    for r in [(1., 2., 3.), (2., 1., 3.), (3., 2., 1.),
              (-1., 2., 3.), (1., -2., 3.), (1., 2., -3.)]:
          assert potential_ee(r) == expected_output

    expected_output = 1./np.sqrt(13.)
    for r in [(0., 2., 3.), (2., 0., 3.), (3., 2., 0.),
              (-0., 2., 3.), (0., -2., 3.), (0., 2., -3.)]:
          assert potential_ee(r) == expected_output

    expected_output = 1./3.
    for r in [(0., 0., 3.), (0., 0., 3.), (3., 0., 0.),
              (-0., 0., 3.), (0., -0., 3.), (0., 0., -3.)]:
          assert potential_ee(r) == expected_output

    expected_output = 5./(np.sqrt(3.) * 6.)
    for r in [(1., 2., 3., 4., 5., 6., 7., 8., 9.)]:
          assert potential_ee(r) == expected_output

    r = (0., 0., 0.)
    assert potential_ee(r) == float("inf")

    print("potential_ee() -> ok")

def test_potential_eN():
    expected_output = -potential_ee((1., 2., 3.))
    R = (0., 0., 0.)
    Z = [1]
    for r in [(1., 2., 3.), (2., 1., 3.), (3., 2., 1.),
              (-1., 2., 3.), (1., -2., 3.), (1., 2., -3.)]:
          assert potential_eN(r, R, Z) == expected_output

    expected_output = -2. * potential_ee((1., 2., 3.))
    R = (0., 0., 0.)
    Z = [2]
    for r in [(1., 2., 3.), (2., 1., 3.), (3., 2., 1.),
              (-1., 2., 3.), (1., -2., 3.), (1., 2., -3.)]:
          assert potential_eN(r, R, Z) == expected_output

    expected_output = -(
            1./np.sqrt(77.) + 1./np.sqrt(14) + np.sqrt(2.)/5. + 2./np.sqrt(5.)
            )
    r = (1., 2., 3., 4., 5., 6.)
    R = (0., 0., 0., 1., 1., 1.)
    Z = [1, 2]
    assert potential_eN(r, R, Z) == expected_output

    r = (0., 0., 0.)
    assert potential_eN(r, R, Z) == -float("inf")

    print("potential_eN() -> ok")

def test_potential_NN():
    expected_output = 2./np.sqrt(3.)
    R = (0., 0., 0., 1., 1., 1.)
    Z = [1, 2]
    assert potential_NN(R, Z) == expected_output

    R = (1., 1., 1.)
    Z = [2]
    expected_output = Z[0] * potential_ee(R)
    assert potential_NN(R, Z) == expected_output

    expected_output = 0.
    R = (0., 0., 0.)
    Z = [1]
    assert potential_NN(R, Z) == expected_output

    print("potential_NN() -> ok")

