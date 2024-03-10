from hamiltonian import phi, d_phi, d2_phi, psi, d_psi, d2_psi, potential_ee, potential_eN, potential_NN, potential, kinetic_N, kinetic_e, kinetic

import numpy as np

def all_debug():
    test_phi()
    test_dphi()
    test_d2phi()
    test_psi()
    test_dpsi()
    test_d2psi()
    test_potential_ee()
    test_potential_eN()
    test_potential_NN()
    test_potential()
    test_kinetic_N()
    test_kinetic_e()
    test_kinetic()

def test_phi():
    # Case when |rij| = 0
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = (a**3/np.pi)**0.5
    assert phi(a, r, R) == expected_output

    # Case when |rij| = 0
    a = 1.2
    r = (1., 1., 1.)
    R = (1., 1., 1.)
    expected_output = (a**3/np.pi)**0.5
    assert phi(a, r, R) == expected_output

    print("phi() -> ok")

def test_dphi():
    # Case when |rij| = 0
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = -float("inf")
    assert d_phi('x', a, r, R) == expected_output

    # Case when Cij = 0, |rij| /= 0
    a = 1.2
    r = (1., 2., 2.)
    R = (1., 0., 0.)
    expected_output = 0.
    assert d_phi('x', a, r, R) == expected_output

    # Case when Cij = 1 <- xi - Xj = 1 and |rij| = 1
    a = 1.2
    r = (2., 0., 0.)
    R = (1., 0., 0.)
    expected_output = -a * phi(a, r, R)
    assert d_phi('x', a, r, R) == expected_output

    # Test with numerical derivative
    x = np.linspace(-5.,5,100)
    dx = x[1]-x[0]
    y = []
    yp = []
    for i in range(len(x)):
        r = (x[i], 1., 0.)
        y.append(phi(a, r, R))
        yp.append(d_phi('x', a, r, R))
    # compute numerically the derivative as the gradient
    num_yp = np.gradient(y, dx)
    # compare the values with a relative tolerance of 0.1 * num_yp (10%)
    assert np.allclose(yp[1:-1], num_yp[1:-1], rtol=1e-01)

    print("d_phi() -> ok")

def test_d2phi():
    # Check with first derivative
    a = 1.2
    r = (1., 1., 1.)
    R = (0., 0., 0.)
    rij = np.sqrt((r[0] - R[0])**2 + (r[1] - R[1])**2 + (r[2] - R[2])**2)
    Cij = (r[0] - R[0])/rij
    expected_output = -a/rij * phi(a, r, R) - (Cij * (1 + a * rij))/rij * d_phi('x', a, r, R)
    assert np.isclose(d2_phi('x', a, r, R), expected_output, atol=1e-16)

    # Case when |rij| = 0
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = -float("inf")
    assert d2_phi('x', a, r, R) == expected_output

    # Case when Cij = 0, |rij| /= 0
    a = 1.2
    r = (1., 1., 1.)
    R = (1., 0., 0.)
    rij = np.sqrt((r[0] - R[0])**2 + (r[1] - R[1])**2 + (r[2] - R[2])**2)
    expected_output = -a/rij * phi(a, r, R)
    assert d2_phi('x', a, r, R) == expected_output

    # Case when Dij = 1. <- C = 0. and |rij| = 1.
    a = 1.2
    r = (1., 1., 0.)
    R = (1., 0., 0.)
    expected_output = -a * phi(a, r, R)
    assert d2_phi('x', a, r, R) == expected_output

    # Test with numerical derivative
    x = np.linspace(-5.,5,100)
    dx = x[1]-x[0]
    y = []
    y2p = []
    for i in range(len(x)):
        r = (x[i], 1., 0.)
        y.append(phi(a, r, R))
        y2p.append(d2_phi('x', a, r, R))
    # compute numerically the derivative as the gradient
    num_yp = np.gradient(y, dx)
    num_y2p = np.gradient(num_yp, dx)
    # compare the values with a relative tolerance of 0.3 * num_yp (30%)
    # remove first and last points of the gradient
    assert np.allclose(y2p[1:-1], num_y2p[1:-1], rtol=3.0e-01)

    print("d2_phi() -> ok")

def test_psi():
    # Case when |rij| = 0
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = phi(a, r, R)
    assert psi(a, r, R) == expected_output

    # Case for 1 electron and 1 atom and R = 0
    a = 1.2
    r = (1., 2., 3.)
    R = (0., 0., 0.)
    expected_output = phi(a, r, R)
    assert psi(a, r, R) == expected_output

    # Case for 1 electron and 1 atom and R /= 0
    a = 1.2
    r = (1., 2., 3.)
    R = (4., 5., 6.)
    expected_output = phi(a, r, R)
    assert psi(a, r, R) == expected_output

    # Case for 2 electron and 2 atom
    a = 1.2
    r = (1., 2., 3., 4., 5., 6.)
    R = (7., 8., 9., 10., 11., 12.)
    expected_output = ( 
                       phi(a, r[0:3], R[0:3]) + phi(a, r[0:3], R[3:6]) 
                       ) * ( 
                            phi(a, r[3:6], R[0:3]) + phi(a, r[3:6], R[3:6]) 
                       )
    assert psi(a, r, R) == expected_output

    print("psi() -> ok")

def test_dpsi():
    # Case when |rij| = 0
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = float("inf")
    assert d_psi('x', a, r, R) == expected_output

    # Case when |rij| = 0 but ri /= 0 and Rj /= 0
    r = (1., 1., 1.)
    R = (1., 1., 1.)
    expected_output = float("inf")
    assert d_psi('x', a, r, R) == expected_output

    # Case when Cij = 0, |rij| /= 0 (1 electron)
    r = (0., 2., 2.)
    R = (0., 1., 1.)
    expected_output = 0.
    assert d_psi('x', a, r, R) == expected_output

    # Case for 1 electron, check with derivative of phi
    r = (1., 2., 3.)
    R = (4., 5., 6.)
    expected_output = d_phi('x', a, r, R)
    assert d_psi('x', a, r, R) == expected_output

    # Case when Cij = 0, |rij| /= 0 (multiple electrons)
    r = (0., 2., 2., 0., 2., 2., 0., 2., 2., 0., 2., 2.)
    R = (0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1.)
    expected_output = 0.
    assert d_psi('x', a, r, R) == expected_output

    # Test with numerical derivative
    x = np.linspace(-5.,5,100)
    dx = x[1]-x[0]
    y = []
    yp = []
    for i in range(len(x)):
        r = (x[i], 1., 0.)
        y.append(psi(a, r, R))
        yp.append(d_psi('x', a, r, R))
    # compute numerically the derivative as the gradient
    num_yp = np.gradient(y, dx)
    # compare the values with a relative tolerance of 0.1 * num_yp (10%)
    assert np.allclose(yp[1:-1], num_yp[1:-1], rtol=1e-01)

    print("d_psi() -> ok")

def test_d2psi():
    # Case when |rij| = 0 -> Dij = inf
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = float("inf")
    assert d2_psi('x', a, r, R) == expected_output

    # Case when |rij| = 0 -> Dij = inf
    r = (1., 1., 1.)
    R = (1., 1., 1.)
    expected_output = float("inf")
    assert d2_psi('x', a, r, R) == expected_output

    # Case when Cij = 0 -> Dij = 1/rij but |rij| /= 0
    r = (0., 1., 1.)
    R = (0., 0., 0.)
    rij = np.sqrt((r[0] - R[0])**2 + (r[1] - R[1])**2 + (r[2] - R[2])**2)
    expected_output = -a * 1./rij * psi(a, r, R)
    assert d2_psi('x', a, r, R) == expected_output

    # Case when Dij = 1 <- Cij = 0, |rij| = 1
    r = (0., 1., 0.)
    R = (0., 0., 0.)
    expected_output = -a * psi(a, r, R)
    assert d2_psi('x', a, r, R) == expected_output

    # Test with numerical derivative
    x = np.linspace(-5.,5,100)
    dx = x[1]-x[0]
    y = []
    yp = []
    for i in range(len(x)):
        r = (x[i], 1., 0.)
        y.append(psi(a, r, R))
        yp.append(d_psi('x', a, r, R))
    # compute numerically the derivative as the gradient
    num_yp = np.gradient(y, dx)
    # compare the values with a relative tolerance of 0.1 * num_yp (10%)
    assert np.allclose(yp[1:-1], num_yp[1:-1], rtol=1e-01)

    print("d2_psi() -> ok")

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

    # Case when |r| = 0
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

    # Case when |r - R| = 0
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

    # Case when |R| = 0
    expected_output = 0.
    R = (0., 0., 0.)
    Z = [1]
    assert potential_NN(R, Z) == expected_output

    print("potential_NN() -> ok")

def test_potential():
    r = (1., 1., 1.)
    R = (0., 0., 0.)
    Z = [1]
    expected_output = potential_ee(r) + potential_eN(r, R, Z) + potential_NN(R, Z)
    assert potential(r, R, Z) == expected_output

    print("potential() -> ok")

def test_kinetic_N():
    # Born-Oppenheimier approximation
    expected_output = 0.
    assert kinetic_N() == expected_output

    print("kinetic_N() -> ok")

def test_kinetic_e():
    # Case when |rij| = 0
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = float("inf")
    assert kinetic_e(a, r, R) == expected_output

    # Case for 1 electron => K_e = 1/2 * alpha * D_{11}
    a = 1.2
    r = (1., 1., 1.)
    R = (0., 0., 0.)
    rij = np.sqrt((r[0] - R[0])**2 + (r[1] - R[1])**2 + (r[2] - R[2])**2)
    D11 = 3./rij - (1 + a * rij)/rij
    expected_output = 1./2. * a * D11
    assert kinetic_e(a, r, R) == expected_output

    # Case when Dij = 0 <- 1 + a|rij| = 3 <- a = 2/|rij|
    r = (1., 1., 1.)
    R = (0., 0., 0.)
    rij = np.sqrt((r[0] - R[0])**2 + (r[1] - R[1])**2 + (r[2] - R[2])**2)
    a = 2./rij
    expected_output = 0.
    assert kinetic_e(a, r, R) == expected_output

    print("kinetic_e() -> ok")

def test_kinetic():
    a = 1.2
    r = (1., 1., 1.)
    R = (0., 0., 0.)
    expected_output = kinetic_e(a, r, R) + kinetic_N()
    assert kinetic(a, r, R) == expected_output

    print("kinetic() -> ok")

