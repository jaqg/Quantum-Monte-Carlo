import numpy as np

# Electron-electron potential
def potential_ee(r):
    # r: electron coordinates vector
    # r -> (x1, y1, z1, x2, y2, z2, ..., xn, yn, zn)
    n = int(len(r)/3)  # number of electrons
    pot = 0.

    # If there is one electron, the potential becomes 1/|r|
    if n == 1:
        r_mod = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        # if |r| = 0, 1/|r| -> inf
        if r_mod == 0.:
            return float("inf")
        else:
            return 1./r_mod
    # For more than 1 electron, compute the pair repulsion
    else:
        for i in range(n+1):
            i_ind = 3*i
            for j in range(i+1, n):
                j_ind = 3*j
                rij = np.sqrt(
                        (r[i_ind] - r[j_ind])**2 +      # (xi - xj)^2
                        (r[i_ind+1] - r[j_ind+1])**2 +  # (yi - yj)^2
                        (r[i_ind+2] - r[j_ind+2])**2    # (zi - zj)^2
                        )
                if rij == 0.:
                    return float("inf")
                else:                    
                    pot += 1./rij
        return pot

# Electron-nucleus potential
def potential_eN(r, R, Z):
    # r: electron coordinates vector
    # R: nucleus coordinates vector
    # Z: nucleus charge
    # r -> (x1, y1, z1, x2, y2, z2, ..., xn, yn, zn)
    # R -> (x1, y1, z1, x2, y2, z2, ..., xm, ym, zm)
    # Z -> (z1, z2, ..., zm)
    n = int(len(r)/3)  # number of electrons
    m = int(len(R)/3)  # number of nucleus

    # Check that the number of Z values is the same as number of nucleus
    if len(Z) != m:
        raise ValueError('The number of Z values must be the same as the\
                         number of nucleus')
        
    pot = 0.
    for i in range(n):
        i_ind = 3*i
        for j in range(m):
            j_ind = 3*j
            rij = np.sqrt(
                    (r[i_ind]   - R[j_ind])**2 +    # (xi - xj)^2
                    (r[i_ind+1] - R[j_ind+1])**2 +  # (yi - yj)^2
                    (r[i_ind+2] - R[j_ind+2])**2    # (zi - zj)^2
                    )
            if rij == 0.:
                return -float("inf")
            else:                    
                pot += float(Z[j])/rij
    return -pot

# Nucleus-nucleus potential
def potential_NN(R, Z):
    # R: nucleus coordinates vector
    # Z: nucleus charge
    # R -> (x1, y1, z1, x2, y2, z2, ..., xm, ym, zm)
    # Z -> (z1, z2, ..., zm)

    m = int(len(R)/3)  # number of nucleus

    # Check that the number of Z values is the same as number of nucleus
    if len(Z) != m:
        raise ValueError('The number of Z values must be the same as the\
                         number of nucleus')

    pot = 0.

    # If there is one nucleus, the potential becomes Z/|R|
    if m == 1:
        R_mod = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2)
        # if |R| = 0, take potential = 0.
        if R_mod == 0.:
            return 0.
        else:
            return Z[0]/R_mod
    # For more than 1 nucleus, compute the pair repulsion
    else:
        for i in range(m+1):
            i_ind = 3*i
            for j in range(i+1, m):
                j_ind = 3*j
                Rij = np.sqrt(
                        (R[i_ind] - R[j_ind])**2 +      # (xi - xj)^2
                        (R[i_ind+1] - R[j_ind+1])**2 +  # (yi - yj)^2
                        (R[i_ind+2] - R[j_ind+2])**2    # (zi - zj)^2
                        )
                if Rij == 0.:
                    return float("inf")
                else:                    
                    pot += Z[i] * Z[j]/Rij
        return pot

# Total potential energy
def potential(r, R, Z):
    return potential_ee(r) + potential_eN(r, R, Z) + potential_NN(R, Z)

# Kinetic energy of the nuclei
def kinetic_N():
    # Born-Oppenheimer approximation is considered
    return 0.
