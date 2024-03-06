from debug import all_debug
from hamiltonian import phi, d_phi, d2_phi, psi, d_psi, d2_psi
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

phi_r = phi(a, r, R)
dphi = d_phi('x', a, r, R)
print('phi =', phi_r, 'dphi =', dphi)

psi_r = psi(a, r, R)
dpsi = d_psi('x', a, r, R)
print('psi =', psi_r, 'dpsi =', dpsi)

# -----------------------------------------------------------------------------
# Read Input
section('Numerical derivatives')
# -----------------------------------------------------------------------------
R = (-2., 0., 0.)
x = np.linspace(-5.,5,100)
dx = x[1]-x[0]
nphi = []
nphip = []
nphip2 = []
npsi = []
npsip = []
npsip2 = []
for i in range(len(x)):
    r = (x[i], 1., 0.)
    nphi.append(phi(a, r, R))
    nphip.append(d_phi('x', a, r, R))
    nphip2.append(d2_phi('x', a, r, R))
    npsi.append(psi(a, r, R))
    npsip.append(d_psi('x', a, r, R))
    npsip2.append(d2_psi('x', a, r, R))

# gradient
dnphidx = np.gradient(nphi, dx)
dnphidx2 = np.gradient(dnphidx, dx)

dnpsidx = np.gradient(npsi, dx)
dnpsidx2 = np.gradient(dnpsidx, dx)

# finite difference
ndx=np.diff(x,1)
nxfirst=0.5*(x[:-1]+x[1:])
ndxfirst=np.diff(nxfirst,1)
nxsecond=0.5*(nxfirst[:-1]+nxfirst[1:])

ndnphi=np.diff(nphi,1)
nphifirst=ndnphi/ndx
ndnphifirst=np.diff(nphifirst,1)
nphisecond=ndnphifirst/ndxfirst

ndnpsi=np.diff(npsi,1)
npsifirst=ndnpsi/ndx
ndnpsifirst=np.diff(npsifirst,1)
npsisecond=ndnpsifirst/ndxfirst

# plot
fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

ax.plot(x, nphi, label=r'$\phi$')
ax.plot(x, nphip, label=r'$\partial_x \phi$')
ax.plot(x, nphip2, label=r'$\partial_x^2 \phi$')
ax.plot(x, dnphidx, label=r'numerical gradient')
ax.plot(x, dnphidx2, label=r'num. second derivative')
ax.plot(nxfirst, nphifirst, label=r'numerical gradient fin. diff.')
ax.plot(nxsecond, nphisecond, label=r'num. 2nd fin. diff.')

# ax.plot(x, npsi, label=r'$\psi$')
# ax.plot(x, npsip, label=r'$\partial_x \psi$')
# ax.plot(x, npsip2, label=r'$\partial_x^2 \psi$')
# ax.plot(x, dnpsidx, label=r'numerical gradient')
# ax.plot(x, dnpsidx2, label=r'num. second derivative')
# ax.plot(nxfirst, nnpsifirst, label=r'numerical gradient fin. diff.')
# ax.plot(nxsecond, nnpsisecond, label=r'num. 2nd fin. diff.')

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
x, y, z, X, Y, Z, a, pi, rij, r, r2, phi = symbols('x y z X Y Z a pi rij r r2 phi')
rij = sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
f = (a**3/pi)**0.5 * exp(-a * rij)

df = diff(f, x)
df = df.subs(rij, r)
df = df.subs((a**3/pi)**0.5 * exp(-a * r), phi)
print('1st der:', df)

d2f = diff(f, x, x)
d2f = d2f.subs(rij, r)
d2f = d2f.subs((X - x)**2 + (Y - y)**2 + (Z - z)**2, r2)
d2f = d2f.subs(sqrt(r2), r)
d2f = d2f.subs((a**3/pi)**0.5 * exp(-a * r), phi)
d2f = simplify(d2f)
print('2nd der:', d2f)

# -----------------------------------------------------------------------------
end()
