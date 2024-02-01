from hydrogen import potential

# Test function
def test_potential():
    expected_output = -1./15.
    for r in [( 2., 5., 14.), (5., 14., 2.), 
              (-2., 5.,-14.), (5.,-14.,-2.), 
              ( 0., 9.,-12.), (9.,-12., 0.)]:
          assert potential(r) == expected_output

    r = (0., 0., 0.)
    assert potential(r) == -float("inf")

    print("potential ok")
