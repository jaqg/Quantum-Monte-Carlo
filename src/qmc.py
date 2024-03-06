from debug import all_debug
from hamiltonian import psi, d_psi, d2_psi
from IO import title, section, end

import numpy as np
from sympy import *
import matplotlib.pyplot as plt

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
psi_r = psi(a, r, R)
dpsi = d_psi('x', a, r, R)
print('psi =', psi_r, 'dpsi =', dpsi)

# -----------------------------------------------------------------------------
# Read Input
section('Numerical derivatives')
# -----------------------------------------------------------------------------
R = (-2., 0., 0., 2., 0., 0.)
x = np.linspace(-5.,5,100)
dx = x[1]-x[0]
y = []
yp = []
yp2 = []
for i in range(len(x)):
    r = (x[i], 1., 0., x[i], 1., 0.)
    y.append(psi(a, r, R))
    yp.append(d_psi('x', a, r, R))
    yp2.append(d2_psi('x', a, r, R))

dydx = np.gradient(y, dx)
dydx2 = np.gradient(dydx, dx)

ndy=np.diff(y,1)
ndx=np.diff(x,1)
nyfirst=ndy/ndx
nxfirst=0.5*(x[:-1]+x[1:])
ndyfirst=np.diff(nyfirst,1)
ndxfirst=np.diff(nxfirst,1)
nysecond=ndyfirst/ndxfirst
nxsecond=0.5*(nxfirst[:-1]+nxfirst[1:])

# ypprime = 2

# print('gradient:\n', dydx,'\n')
# print('d_psi():\n', yp,'\n')
# print('gradient - d_psi():\n', dydx - yp,'\n')
# print('relative in base 1:\n', (dydx - yp)/yp,'\n')
# print('dydx2: ', dydx2)
# print('gradient2 - ypprime: ', dydx2 - ypprime)

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

ax.plot(x, y, label=r'$\psi$')
ax.plot(x, yp, label=r'$\partial_x \psi$')
ax.plot(x, yp2, label=r'$\partial_x^2 \psi$')
ax.plot(x, dydx, label=r'numerical gradient')
ax.plot(x, dydx2, label=r'num. second derivative')
ax.plot(nxfirst, nyfirst, label=r'numerical gradient fin. diff.')
ax.plot(nxsecond, nysecond, label=r'num. 2nd fin. diff.')

for i in range(int(len(R)/3)):
    ax.vlines(x=R[3*i], ymin=-1., ymax=1., color='k', linestyle='--')
ax.hlines(y=0., xmin=-5., xmax=5., color='k', linestyle='--')

ax.set(
        xlabel=r'x',
        ylabel=r'y'
        )
ax.legend()

# plt.show()

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
