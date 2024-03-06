import numpy as np
#
# Functions to describe the system: basis functions, wavefunction and hamiltonian
# For mathematical details see the documentation
#

# Slater determinants as AO basis
def phi(a, r, R):
    # 1s Slater type function phi_{ij}(r) = (a^3/pi)^{1/2} exp(-a |r_i - R_j|)
    # a: Slater orbital exponent
    # r: i-th electron coordinates vector
    # R: j-th nucleus coordinates vector
    # a -> float
    # r -> (xi, yi, zi)
    # R -> (xj, yj, zj)

    rij2 = (r[0] - R[0])**2 + (r[1] - R[1])**2 + (r[2] - R[2])**2
    if rij2 < 0:
        return 0
    else:
        rij = np.sqrt(rij2)
        return (a**3/np.pi)**0.5 * np.exp(-a * rij)

# First derivative of the basis functions
def d_phi(llambda, a, r, R):
    # First derivative of the 1s Slater type function
    # partial_{lambda} phi(r_i; R_j) for lambda = {x, y, z}

    # llambda: coordinate which the derivative is taken
    # a: Slater orbital exponent
    # r: i-th electron coordinates vector
    # R: j-th nucleus coordinates vector
    # llambda -> str
    # a -> float
    # r -> (xi, yi, zi)
    # R -> (xj, yj, zj)

    if llambda == 'x':
        lambda_ind = 0
    elif llambda == 'y':
        lambda_ind = 1
    elif llambda == 'z':
        lambda_ind = 2
    else:
        raise ValueError('lambda must be x, y or z')

    rij2 = (r[0] - R[0])**2 + (r[1] - R[1])**2 + (r[2] - R[2])**2
    if rij2 < 0:
        return 0
    else:
        rij = np.sqrt(rij2)
        if rij == 0.:
            return -float("inf")
        else:                    
            Cij = (r[lambda_ind] - R[lambda_ind])/rij
            return - a * Cij * phi(a, r, R)

# Second derivative of the basis functions
def d2_phi(llambda, a, r, R):
    # Second derivative of the 1s Slater type function
    # partial_{lambda}^2 phi(r_i; R_j) for lambda = {x, y, z}

    # llambda: coordinate which the derivative is taken
    # a: Slater orbital exponent
    # r: i-th electron coordinates vector
    # R: j-th nucleus coordinates vector
    # llambda -> str
    # a -> float
    # r -> (xi, yi, zi)
    # R -> (xj, yj, zj)

    if llambda == 'x':
        lambda_ind = 0
    elif llambda == 'y':
        lambda_ind = 1
    elif llambda == 'z':
        lambda_ind = 2
    else:
        raise ValueError('lambda must be x, y or z')

    rij2 = (r[0] - R[0])**2 + (r[1] - R[1])**2 + (r[2] - R[2])**2
    if rij2 < 0:
        return 0
    else:
        rij = np.sqrt(rij2)
        if rij == 0.:
            return -float("inf")
        else:                    
            Cij = (r[lambda_ind] - R[lambda_ind])/rij
            Dij = 1./rij - (Cij**2 * (1. + a * rij))/rij
            return - a * Dij * phi(a, r, R)

# Wavefunction
def psi(a, r, R):
    # Wavefunction as 1s Slater type functions centered in each atom
    # a: Slater orbital exponent
    # r: electron coordinates vector
    # R: nucleus coordinates vector
    # a -> float
    # r -> (x1, y1, z1, x2, y2, z2, ..., xn, yn, zn)
    # R -> (x1, y1, z1, x2, y2, z2, ..., xm, ym, zm)

    n = int(len(r)/3)  # number of electrons
    m = int(len(R)/3)  # number of nucleus

    psi = 1.
    for i in range(n):
        ri = r[3*i:3*i+3]
        suma = 0.
        for j in range(m):
            Rj = R[3*j:3*j+3]
            suma += phi(a, ri, Rj)
        psi *= suma
    
    return psi

# First derivative of the wavefunction
def d_psi(llambda, a, r, R):
    # First derivative of the wavefunction: partial_{lambda} psi(r_i; R_j)
    # for lambda = {x, y, z}
    #
    # llambda: coordinate which the derivative is taken
    # a: Slater orbital exponent
    # r: i-th electron coordinates vector
    # R: nucleus coordinates vector
    # llambda -> str
    # a -> float
    # r -> (xi, yi, zi)
    # R -> (xj, yj, zj)

    if llambda == 'x':
        lambda_ind = 0
    elif llambda == 'y':
        lambda_ind = 1
    elif llambda == 'z':
        lambda_ind = 2
    else:
        raise ValueError('lambda must be x, y or z')

    n = int(len(r)/3)  # number of electrons
    m = int(len(R)/3)  # number of nucleus
        
    suma = 0.
    for i in range(n):
        i_ind = 3*i
        ri = r[i_ind:i_ind+3]
        nominator = 0.
        denominator = 0.
        for j in range(m):
            j_ind = 3*j
            Rj = R[j_ind:j_ind+3]
            rij2 = ((ri[0] - Rj[0])**2 +  # (xi - Xj)^2
                    (ri[1] - Rj[1])**2 +  # (yi - Yj)^2
                    (ri[2] - Rj[2])**2    # (zi - Zj)^2
                   )
            if rij2 < 0.:
                return 0
            else:
                rij = np.sqrt(rij2)
                if rij == 0.:
                    return float("inf")
                else:                    
                    Cij = (r[lambda_ind] - R[lambda_ind])/rij
                    nominator += Cij * phi(a, ri, Rj)
            denominator += phi(a, ri, Rj)
        if denominator == 0.:
            return float("inf")
        else:
            suma += nominator/denominator
    return -a * suma * psi(a, r, R)

# Second derivative of the wavefunction
def d2_psi(llambda, a, r, R):
    # Second derivative of the wavefunction: partial_{lambda}^2 psi(r_i; R_j)
    # for lambda = {x, y, z}
    #
    # lambda: coordinate which the derivative is taken
    # a: Slater orbital exponent
    # r: i-th electron coordinates vector
    # R: nucleus coordinates vector
    # lambda -> str
    # a -> float
    # r -> (xi, yi, zi)
    # R -> (xj, yj, zj)
    if llambda == 'x':
        lambda_ind = 0
    elif llambda == 'y':
        lambda_ind = 1
    elif llambda == 'z':
        lambda_ind = 2
    else:
        raise ValueError('lambda must be x, y or z')

    n = int(len(r)/3)  # number of electrons
    m = int(len(R)/3)  # number of nucleus
        
    suma = 0.
    for i in range(n):
        i_ind = 3*i
        ri = r[i_ind:i_ind+3]
        nominator = 0.
        denominator = 0.
        for j in range(m):
            j_ind = 3*j
            Rj = R[j_ind:j_ind+3]
            rij2 = ((ri[0] - Rj[0])**2 +  # (xi - Xj)^2
                    (ri[1] - Rj[1])**2 +  # (yi - Yj)^2
                    (ri[2] - Rj[2])**2    # (zi - Zj)^2
                   )
            if rij2 < 0.:
                return 0
            else:
                rij = np.sqrt(rij2)
                if rij == 0.:
                    return float("inf")
                else:                    
                    Dij = 1./rij - ( (a * rij + 1.) * (ri[lambda_ind] - Rj[lambda_ind])**2 )/(rij**3)
                    nominator += Dij * phi(a, ri, Rj)
            denominator += phi(a, ri, Rj)
        if denominator == 0.:
            return float("inf")
        else:
            suma += nominator/denominator
    return -a * suma * psi(a, r, R)

# Electron-electron potential
def potential_ee(r):
    # r: electron coordinates vector
    # r -> (x1, y1, z1, x2, y2, z2, ..., xn, yn, zn)
    n = int(len(r)/3)  # number of electrons
    pot = 0.

    # If there is one electron, the potential becomes 1/|r|
    if n == 1:
        r2 = r[0]**2 + r[1]**2 + r[2]**2
        if r2 < 0.:
            return 0
        else:
            r_mod = np.sqrt(r2)
            # if |r| = 0, 1/|r| -> inf
            if r_mod == 0.:
                return float("inf")
            else:
                return 1./r_mod
    # For more than 1 electron, compute the pair repulsion
    else:
        for i in range(n+1):
            i_ind = 3*i
            ri = r[i_ind:i_ind+3]
            for j in range(i+1, n):
                j_ind = 3*j
                rj = r[j_ind:j_ind+3]
                rij2 = ((ri[0] - rj[0])**2 +  # (xi - Xj)^2
                        (ri[1] - rj[1])**2 +  # (yi - Yj)^2
                        (ri[2] - rj[2])**2    # (zi - Zj)^2
                       )
                if rij2 < 0.:
                    return 0
                else:
                    rij = np.sqrt(rij2)
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
        ri = r[i_ind:i_ind+3]
        for j in range(m):
            j_ind = 3*j
            Rj = R[j_ind:j_ind+3]
            rij2 = ((ri[0] - Rj[0])**2 +  # (xi - Xj)^2
                    (ri[1] - Rj[1])**2 +  # (yi - Yj)^2
                    (ri[2] - Rj[2])**2    # (zi - Zj)^2
                   )
            if rij2 < 0.:
                return 0
            else:
                rij = np.sqrt(rij2)
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
        R2 = R[0]**2 + R[1]**2 + R[2]**2
        if R2 < 0.:
            return 0
        else:
            R_mod = np.sqrt(R2)
            # if |R| = 0, take potential = 0.
            if R_mod == 0.:
                return 0.
            else:
                return Z[0]/R_mod
    # For more than 1 nucleus, compute the pair repulsion
    else:
        for i in range(m+1):
            i_ind = 3*i
            Ri = R[i_ind:i_ind+3]
            for j in range(i+1, m):
                j_ind = 3*j
                Rj = R[j_ind:j_ind+3]
                Rij2 = ((Ri[0] - Rj[0])**2 +  # (xi - Xj)^2
                        (Ri[1] - Rj[1])**2 +  # (yi - Yj)^2
                        (Ri[2] - Rj[2])**2    # (zi - Zj)^2
                       )
                if Rij2 < 0.:
                    return 0
                else:
                    Rij = np.sqrt(Rij2)
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
