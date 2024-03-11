from hamiltonian import phi, d_phi, d2_phi, psi, d_psi, d2_psi
from IO import section

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import os

style_file = 'mine.mplstyle'
style_file = os.path.dirname(__file__)+'/{}'.format(style_file)
plt.style.use(style_file)

# -----------------------------------------------------------------------------
section('Analitical derivatives')
# -----------------------------------------------------------------------------
x, y, z, X, Y, Z, a, pi, rij, r, r2, phi_ij = sp.symbols('x y z X Y Z a pi rij r r2 phi_ij')
rij = sp.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
f = (a**3/pi)**0.5 * sp.exp(-a * rij)

df = sp.diff(f, x)
df = df.subs(rij, r)
df = df.subs((a**3/pi)**0.5 * sp.exp(-a * r), phi_ij)
print('1st der:', df)

d2f = sp.diff(f, x, x)
d2f = d2f.subs(rij, r)
d2f = d2f.subs((X - x)**2 + (Y - y)**2 + (Z - z)**2, r2)
d2f = d2f.subs(sp.sqrt(r2), r)
d2f = d2f.subs((a**3/pi)**0.5 * sp.exp(-a * r), phi_ij)
d2f = sp.simplify(d2f)
print('2nd der:', d2f)

# Remove the symbols
symbolsToDelete = ('x', 'y', 'z', 'X', 'Y', 'Z', 'a', 'pi', 'rij', 'r', 'r2', 'phi_ij')

for sym in symbolsToDelete:
    del globals()[sym]

# -----------------------------------------------------------------------------
section('Numerical derivatives')
print('Computing numerical derivatives: gradient & finite difference...')
# -----------------------------------------------------------------------------
# Phi for 1 electron & 1 atom
a = 1.2
R = (0., 0., 0.)
x = np.linspace(-5.,5,100)
dx = x[1]-x[0]
nphi = []
nphip = []
nphip2 = []
for i in range(len(x)):
    r = (x[i], 1., 0.)
    nphi.append(phi(a, r, R))
    nphip.append(d_phi('x', a, r, R))
    nphip2.append(d2_phi('x', a, r, R))

# gradient
dnphidx = np.gradient(nphi, dx)
dnphidx2 = np.gradient(dnphidx, dx)

# finite difference
ndx=np.diff(x,1)
nxfirst=0.5*(x[:-1]+x[1:])
ndxfirst=np.diff(nxfirst,1)
nxsecond=0.5*(nxfirst[:-1]+nxfirst[1:])

ndnphi=np.diff(nphi,1)
nphifirst=ndnphi/ndx
ndnphifirst=np.diff(nphifirst,1)
nphisecond=ndnphifirst/ndxfirst

# Psi for 3 electrons & 3 atoms
R = (-2., 0., 0., 0., 0., 0., 2., 0., 0.)
npsi = []
npsip = []
npsip2 = []
for i in range(len(x)):
    r = (R[0]+x[i], 1., 0., R[3]+x[i], 1., 0., R[6]+x[i], 1., 0.)
    npsi.append(psi(a, r, R))
    npsip.append(d_psi('x', a, r, R))
    npsip2.append(d2_psi('x', a, r, R))

# gradient
dnpsidx = np.gradient(npsi, dx)
dnpsidx2 = np.gradient(dnpsidx, dx)

# finite difference
ndnpsi=np.diff(npsi,1)
npsifirst=ndnpsi/ndx
ndnpsifirst=np.diff(npsifirst,1)
npsisecond=ndnpsifirst/ndxfirst

# -----------------------------------------------------------------------------
section('Plots')
# -----------------------------------------------------------------------------
print('Making plots...')

# Main figure
fig, axs = plt.subplots(1, 2, figsize=(7.2*1.5, 4.45))
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

for ax in axs:
    for i in range(int(len(R)/3)):
        ax.vlines(x=R[3*i], ymin=-1., ymax=1., color='k', linestyle='-', lw=1.)
    ax.hlines(y=0., xmin=-6., xmax=6., color='k', linestyle='-', lw=1)

ax = axs[0]
ax.plot(x, nphi, label=r'$\phi$')
ax.plot(x, nphip, label=r'$\partial_x \phi$')
ax.plot(x, nphip2, label=r'$\partial_x^2 \phi$')
ax.plot(x, dnphidx, label=r'num. $\partial_x \phi$', ls='--')
ax.plot(x, dnphidx2, label=r'num. $\partial_x^2 \phi$', ls='--')
ax.plot(nxfirst, nphifirst, label=r'fin. diff. $\partial_x \phi$', ls='-.')
ax.plot(nxsecond, nphisecond, label=r'fin. diff. $\partial_x^2 \phi$', ls='-.')

# ax.set_xlim(-5.,5.)
ax.set_ylim(-.3,.3)
ax.set(title=r'$\phi\left( \mathbf{r}; \mathbf{R} \right)$')

ax = axs[1]
ax.plot(x, npsi, label=r'$\psi$')
ax.plot(x, npsip, label=r'$\partial_x \psi$')
ax.plot(x, npsip2, label=r'$\partial_x^2 \psi$')
ax.plot(x, dnpsidx, label=r'num. $\partial_x \psi$', ls='--')
ax.plot(x, dnpsidx2, label=r'num. $\partial_x^2 \psi$', ls='--')
ax.plot(nxfirst, npsifirst, label=r'fin. diff. $\partial_x \psi$', ls='-.')
ax.plot(nxsecond, npsisecond, label=r'fin. diff. $\partial_x^2 \psi$', ls='-.')

# ax.set_xlim(-6.,6.)
ax.set_ylim(-.06,.04)
ax.set(title=r'$\psi\left( \mathbf{r}; \mathbf{R} \right)$')

# Draw circles for nucleus & electrons
for ax in axs:
    # Circles for the electrons
    circ_size = 8.
    for i in range(int(len(r)/3)):
        if i == 0:
            ax.plot(r[3*i], r[3*i+1], 'o', color='r', fillstyle='full',
                    markersize=circ_size, label=r'Electrons')
        else:
            ax.plot(r[3*i], r[3*i+1], 'o', color='r', fillstyle='full', markersize=circ_size)
            

    # Circles for the nucleus
    for i in range(int(len(R)/3)):
        if i == 0:
            ax.plot(R[3*i], R[3*i+1], 'o', color='k',  fillstyle='full',
                    markersize=circ_size, label=r'Nucleus')
        else:
            ax.plot(R[3*i], R[3*i+1], 'o', color='k',  fillstyle='full', markersize=circ_size)
            
    # Common settings
    ax.set(
            xlabel=r'$x$',
            ylabel=r'$y$'
            )
    ax.legend(loc='best')

# plt.show()

print('Saving plots in ./plots')
nombre_grafica = os.path.basename(__file__).replace(".py", ".pdf")
dir = os.path.dirname(os.path.abspath(__file__)).replace("src", "plots")
plt.savefig(dir+"/"+nombre_grafica, transparent='True', bbox_inches='tight')
# -----------------------------------------------------------------------------
