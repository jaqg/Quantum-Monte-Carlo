from hamiltonian import phi, psi, d_psi, d2_psi, potential_ee, potential_eN, potential_NN

import numpy as np

def all_debug():
    test_phi()
    test_psi()
    test_dpsi()
    test_d2psi()
    test_potential_ee()
    test_potential_eN()
    test_potential_NN()

def test_phi():
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = (a**3/np.pi)**0.5
    assert phi(a, r, R) == expected_output

    a = 1.2
    r = (1., 1., 1.)
    R = (1., 1., 1.)
    expected_output = (a**3/np.pi)**0.5
    assert phi(a, r, R) == expected_output

    print("phi() -> ok")

def test_psi():
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = phi(a, r, R)
    assert psi(a, r, R) == expected_output

    a = 1.2
    r = (1., 2., 3.)
    R = (0., 0., 0.)
    expected_output = phi(a, r, R)
    assert psi(a, r, R) == expected_output

    a = 1.2
    r = (1., 2., 3.)
    R = (4., 5., 6.)
    expected_output = phi(a, r, R)
    assert psi(a, r, R) == expected_output

    a = 1.2
    r = (1., 2., 3., 4., 5., 6.)
    R = (1., 2., 3., 4., 5., 6.)
    expected_output = ( 
                       phi(a, r[0:3], R[0:3]) + phi(a, r[0:3], R[3:6]) 
                       ) * ( 
                            phi(a, r[3:6], R[0:3]) + phi(a, r[3:6], R[3:6]) 
                       )
    assert psi(a, r, R) == expected_output

    print("psi() -> ok")

def test_dpsi():
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = float("inf")
    assert d_psi('x', a, r, R) == expected_output

    r = (1., 1., 1.)
    R = (1., 1., 1.)
    expected_output = float("inf")
    assert d_psi('x', a, r, R) == expected_output

    r = (0., 2., 2.)
    R = (0., 1., 1.)
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
    a = 1.2
    r = (0., 0., 0.)
    R = (0., 0., 0.)
    expected_output = float("inf")
    assert d2_psi('x', a, r, R) == expected_output

    r = (1., 1., 1.)
    R = (1., 1., 1.)
    expected_output = float("inf")
    assert d2_psi('x', a, r, R) == expected_output

    r = (0., 2., 2.)
    R = (0., 1., 1.)
    rR = np.sqrt((r[0]-R[0])**2 + (r[1]-R[1])**2 + (r[2]-R[2])**2)
    expected_output = -a/rR * phi(a, r, R)
    assert d2_psi('x', a, r, R) == expected_output

    r = (1., 2., 2.)
    R = (1., 1., 1.)
    rR = np.sqrt((r[0]-R[0])**2 + (r[1]-R[1])**2 + (r[2]-R[2])**2)
    expected_output = -a/rR * phi(a, r, R)
    assert d2_psi('x', a, r, R) == expected_output

    # # Test with numerical derivative
    # x = np.linspace(-5.,5,100)
    # dx = x[1]-x[0]
    # y = []
    # yp = []
    # for i in range(len(x)):
    #     r = (x[i], 1., 0.)
    #     y.append(psi(a, r, R))
    #     yp.append(d_psi('x', a, r, R))
    # # compute numerically the derivative as the gradient
    # num_yp = np.gradient(y, dx)
    # # compare the values with a relative tolerance of 0.1 * num_yp (10%)
    # assert np.allclose(yp[1:-1], num_yp[1:-1], rtol=1e-01)

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

