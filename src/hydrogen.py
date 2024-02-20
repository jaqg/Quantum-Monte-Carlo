import numpy as np

# Wavefunction for the hydrogen atom
def psi(a, r):
    d = r[0]**2 + r[1]**2 + r[2]**2
    if d < 0:
        return 0
    else:
        sqr = np.sqrt(d)
        return np.exp(-a * sqr)

# Potential
def potential(r):
    d = r[0]**2 + r[1]**2 + r[2]**2
    if d < 0:
        return 0
    else:
        sqr = np.sqrt(d)
        if sqr == 0:
            # print("potential at r=0 diverges")
            return -float("inf")
        return -1./sqr

# Kinetic energy
def kinetic(a, r):
    d = r[0]**2 + r[1]**2 + r[2]**2
    if d < 0:
        return 0
    else:
        sqr = np.sqrt(d)
        if sqr > 0:
            di = 1./sqr
        else:
            # print("Warning: kinetic energy diverges at r=0")
            di = float("inf")

        return -1./2. * (a**2 - 2*a*di)

# Local energy
def e_loc(a, r):
    return kinetic(a, r) + potential(r)

# Numerically find value of a
# def find_a():
#     eloc = []
#     for i in range(100):
#         # Guess for a
#         a = random.random()
#
#         # Compute the local energy for the given value of a
#         eloc.append(e_loc(a, r))
#
# # Compute the mean value of the local energy
#     mean_eloc = sum(eloc)/len(eloc)
#     print("Mean local energy:", mean_eloc)
#
# # Make a new iterative process to find the value of a that converges a into
# # mean_eloc
#
# # Guess for a
#     a = random.random()
#     for i in range(100):
#         # Compute the local energy for the given value of a
#         local_energy = e_loc(a, r)
#
#         # Compare the value with the mean value found before
#         delta = local_energy - mean_eloc
#         # and update a
#         if abs(delta) > 0.:
#             a = a + a/10.
#         else:
#             a = a - a/10.
#
#         print("delta: ", delta, "a: ", a, "local_energy: ", local_energy)
#     return a
