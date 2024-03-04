from debug import all_debug
from hamiltonian import phi, d_phi
from IO import title, section, end

import numpy as np
from sympy import *

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
section('Hola')
# -----------------------------------------------------------------------------
a = 1.2
r = (1., 2., 3.)
R = (0., 0., 0.)
phi_r = phi(a, r, R)
dphi = d_phi('x', a, r, R)
print('phi =', phi_r, 'dphi =', dphi)

# -----------------------------------------------------------------------------
# Read Input
section('Numerical derivatives')
# -----------------------------------------------------------------------------
x = np.linspace(0,1,100)
dx = x[1]-x[0]
y = []
yp = []
for i in range(len(x)):
    r = (x[i], x[i]**2, x[i]**3)
    y.append(phi(a, r, R))
    yp.append(d_phi('x', a, r, R))

dydx = np.gradient(y, dx)
# dydx2 = np.gradient(dydx, dx)
# ypprime = 2

print('gradient:\n', dydx,'\n')
print('d_phi():\n', yp,'\n')
print('gradient - d_phi():\n', dydx - yp,'\n')
print('relative in base 1:\n', (dydx - yp)/yp,'\n')
# print('dydx2: ', dydx2)
# print('gradient2 - ypprime: ', dydx2 - ypprime)

# -----------------------------------------------------------------------------
# Read Input
section('Analitical derivatives')
# -----------------------------------------------------------------------------
# x, y, z, a, pi, r, phi = symbols('x y z a pi r phi')
# d = sqrt(x**2 + y**2 + z**2)
# f = (a**3/pi)**0.5 * exp(-a * d)
# df = diff(f, x)
# df = df.subs(d, r)
# df = df.subs((a**3/pi)**0.5*exp(-a*r), phi)
# d2f = diff(f, x, x)
# d2f = d2f.subs(d, r)
# d2f = d2f.subs((a**3/pi)**0.5*exp(-a*r), phi)
# print('1st der:', df)
# print('2nd der:', d2f)

# -----------------------------------------------------------------------------
end()
