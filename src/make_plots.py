from IO import section
from plots import plot_eloc_a, plot_eloc_xcoord, plot_psi_xcoord

# -----------------------------------------------------------------------------
# Plot the local energy along the x axis
section('Plots')
# -----------------------------------------------------------------------------
r = (1., 1., 1.)
R = (0., 0., 0.)
Z = [1]

# Store the desired values of a in a list
a_list = [0.1, 0.2, 0.5, 1., 1.5, 2.]
a_opt = 1.2
alim = 2.
xlim = 4.

print('Saving plots in ./plots/')
plot_eloc_a(r, R, Z, alim, 'plots/local-energy-vs-a.pdf')
plot_eloc_xcoord(a_list, R, Z, xlim, 'plots/local-energy.pdf')
plot_psi_xcoord([a_opt], R, Z, xlim, 'plots/E-Psi.pdf')
