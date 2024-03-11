from hamiltonian import e_loc, psi
import matplotlib.pyplot as plt
import numpy as np
import os

style_file = 'mine.mplstyle'
style_file = os.path.dirname(__file__)+'/{}'.format(style_file)
plt.style.use(style_file)

# Plot the local energy vs values of a
def plot_eloc_a(r, R, Z, alim, filename):
    # Set the number of points of the grid and the limit
    npoints = 200

    # Generate (x,y,z) components of the grid
    a = np.linspace(-alim, alim, npoints)

    # Create the figure
    fig, ax = plt.subplots()

    # Plot the curves for each value of a in a_list
    eloc = []
    psia = []
    for aj in a:
        eloc.append(e_loc(aj, r, R, Z))
        psia.append(psi(aj, r, R))
    ax.plot(a, eloc, label=r'$E\left( \mathbf{r}; \mathbf{R}, a \right)$')
    ax.plot(a, psia, label=r'$\Psi\left( \mathbf{r}; \mathbf{R}, a \right)$')
    ax.plot(a, np.conj(psia) * psia,
            label=r'$\left| \Psi\left( \mathbf{r}; \mathbf{R}, a \right) \right|^2$')

    # Set other parameters of the plot
    ax.set(
            # title=r'',
            xlabel=r'$a$',
            ylabel=r'Local energy, $E_L\left( \mathbf{r}; \mathbf{R}, a \right)$ (hartrees)'
            )
    ax.legend()

    plt.savefig(filename, transparent='True', bbox_inches='tight')
    plt.close()

# Plot the local energy
def plot_eloc_xcoord(a_list, R, Z, xlim, filename):
    # Set the number of points of the grid and the limit
    npoints = 200

    # Generate (x,y,z) components of the grid
    rx = np.linspace(-xlim, xlim, npoints)
    ry = np.linspace(0., 0., npoints)
    rz = np.linspace(0., 0., npoints)
    # and store them in an array, r
    r = np.c_[rx, ry, rz]

    # Create the figure
    fig, ax = plt.subplots()

    # Plot the curves for each value of a in a_list
    for i in range(len(a_list)):
        ai = a_list[i]
        x = []
        y = []
        for j in range(len(r)):
            x.append(r[j,0])
            y.append(e_loc(ai, r[j], R, Z))
        ax.plot(x, y, label=r'$a = {}$'.format(ai), lw=1)

    # Set other parameters of the plot
    ax.set(
            # title=r'',
            xlabel=r'$x$ coordinate',
            ylabel=r'Local energy, $E_L\left( \mathbf{r}; \mathbf{R} \right)$ (hartrees)'
            )
    ax.legend(loc='upper right')

    plt.savefig(filename, transparent='True', bbox_inches='tight')
    plt.close()

# Plot E(r), Psi(r), Psi(r)**2
def plot_psi_xcoord(a_list, R, Z, xlim, filename):
    # Set the number of points of the grid and the limit
    npoints = 200

    # Generate (x,y,z) components of the grid
    rx = np.linspace(-xlim, xlim, npoints)
    ry = np.linspace(0., 0., npoints)
    rz = np.linspace(0., 0., npoints)
    # and store them in an array, r
    r = np.c_[rx, ry, rz]

    # Create the figure
    fig, ax = plt.subplots()

    # Plot the curves for each value of a in a_list
    for i in range(len(a_list)):
        ai = a_list[i]
        x = []
        er = []
        psir = []
        for j in range(len(r)):
            x.append(r[j,0])
            er.append(e_loc(ai, r[j], R, Z))
            psir.append(psi(ai, r[j], R))
        ax.plot(x, er, label=r'$E\left( \mathbf{r}; \mathbf{R} \right)$')
        ax.plot(x, psir, label=r'$\Psi\left( \mathbf{r}; \mathbf{R} \right)$')
        ax.plot(x, np.conjugate(psir) * psir,
                label=r'$\left| \Psi\left( \mathbf{r}; \mathbf{R} \right) \right|^2$')
        # sum_i psi**2 E_i / psi**2 for each position


        # Set other parameters of the plot
        ax.set(
                title=r'$a = {}$'.format(ai),
                xlabel=r'$x$ coordinate',
                ylabel=r'a.u.'
                )
        ax.legend()

    plt.savefig(filename, transparent='True', bbox_inches='tight')
    plt.close()
