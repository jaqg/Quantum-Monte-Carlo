from IO import print_table, section, float_format
from hamiltonian import e_loc, phi, psi
import numpy as np
import pandas as pd

# Quantum Monte Carlo class (to store the results for each method)
class QMC:
  def __init__(self, E, s, A, sA):
    self.E = E    # Energy
    self.s = s    # sigma
    self.A = A    # ratio
    self.sA = sA  # sigma_ratio

# Function to compute the average value of an array
def average(a):
    # a: any array
    # a -> array

    n = len(a)
    assert(n>0)

    if n == 1:
        return a[0]
    else:
        return sum(a)/float(n)

# Function to compute the variance of the average
def variance(a):
    # a: any array
    # a -> array

    n = len(a)
    assert(n>0)

    if n == 1:
        return 0.
    else:
        a_av = average(a)
        suma = 0.
        for ai in a:
            suma += (ai - a_av)**2
        return suma/float(len(a) - 1)

# Function to compute the error 
def error(a):
    # a: any array
    # a -> array

    if variance(a) < 0:
        return 0
    else:
        return np.sqrt(variance(a)/float(len(a)))

# Function to compute the drift vecctor
def drift_vector(a, r, R):
    # a: Slater orbital exponent
    # r: electron coordinates vector
    # R: nucleus coordinates vector
    # a -> float
    # r -> (x1, y1, z1, x2, y2, z2, ..., xn, yn, zn)
    # R -> (x1, y1, z1, x2, y2, z2, ..., xm, ym, zm)
    
    n = int(len(r)/3)  # number of electrons
    m = int(len(R)/3)  # number of nucleus

    dvec = np.zeros(len(r))
    for i in range(n):
        i_ind = 3*i
        ri = r[i_ind:i_ind+3]
        numx = 0.
        numy = 0.
        numz = 0.
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
                    return -float("inf")
                else:                    
                    Cxij = (ri[0] - Rj[0])/rij
                    Cyij = (ri[1] - Rj[1])/rij
                    Czij = (ri[2] - Rj[2])/rij
                    numx += Cxij * phi(a, ri, Rj)
                    numy += Cyij * phi(a, ri, Rj)
                    numz += Czij * phi(a, ri, Rj)
                    denominator += phi(a, ri, Rj)

        dvec[i_ind]   = -a * numx/denominator
        dvec[i_ind+1] = -a * numy/denominator
        dvec[i_ind+2] = -a * numz/denominator

    return dvec

# Monte Carlo algorithm
def MC(a, ne, R, Z, nmc, dt):
    # a: Slater orbital exponent
    # ne: number of electrons
    # R: nucleus coordinates vector
    # Z: nucleus charge
    # nmc: number of MC steps
    # dt: size of the L^3 box (value of L) in which the random position is generated
    # a -> float
    # ne -> int
    # R -> (x1, y1, z1, x2, y2, z2, ..., xm, ym, zm)
    # Z -> (z1, z2, ..., zm)
    # nmc -> int
    # dt -> float

    if nmc <= 0:
        raise ValueError('nmc must be >= 1')

    # Initialize variables
    local_energy = 0.
    norm = 0.
    for i in range(1,nmc+1):
        # Generate random position
        # TODO: no hay que centrar el cubo alrededor de cada R_j?
        rn = np.random.uniform(-dt, dt, 3*ne)

        # Evaluate and acumulate the local energy and normalization factor
        wi = np.conjugate(psi(a, rn, R)) * psi(a, rn, R)
        local_energy += wi * e_loc(a, rn, R, Z)
        norm += wi

    return local_energy/norm

# Metropolis symmetric algorithm
def Metropolis_symmetric_MC(a, ne, R, Z, nmc, dt):
    # a: Slater orbital exponent
    # ne: number of electrons
    # R: nucleus coordinates vector
    # Z: nucleus charge
    # nmc: number of MC steps
    # dt: size of the L^3 box (value of L) in which the random position is generated
    # a -> float
    # ne -> int
    # R -> (x1, y1, z1, x2, y2, z2, ..., xm, ym, zm)
    # Z -> (z1, z2, ..., zm)
    # nmc -> int
    # dt -> float

    local_energy = 0.
    n_accept = 0

    rn = np.random.uniform(-dt, dt, 3*ne)
    psin = psi(a, rn, R)
    for i in range(nmc):
        
        # Evaluate the local energy at r_n
        local_energy += e_loc(a, rn, R, Z)

        # Compute new position
        rprime = rn + np.random.uniform(-dt, dt, 3*ne)

        # Evaluate Psi at the new position
        psiprime = psi(a, rprime, R)

        # Compute the ratio
        a_ratio = min(1., (psiprime / psin)**2)

        # Draw an uniform random number
        v = np.random.uniform()

        # If v <= A, accept the move: set r_{n+1} = rprime
        if v <= a_ratio:
            n_accept += 1
            r_np1 = rprime
            psi_np1 = psiprime
        # else, reject the move: set r_{n+1} = rn
        else:
            r_np1 = rn
            psi_np1 = psin

        # Update the position
        rn = r_np1
        psin = psi_np1

    return local_energy/nmc, n_accept/nmc

# Metropolis generalized algorithm
def Metropolis_generalized_MC(a, ne, R, Z, nmc, dt):
    # a: Slater orbital exponent
    # ne: number of electrons
    # R: nucleus coordinates vector
    # Z: nucleus charge
    # nmc: number of MC steps
    # dt: variance of the Gaussian random numbers
    # a -> float
    # ne -> int
    # R -> (x1, y1, z1, x2, y2, z2, ..., xm, ym, zm)
    # Z -> (z1, z2, ..., zm)
    # nmc -> int
    # dt -> float

    local_energy = 0.
    n_accept = 0

    rn = np.random.normal(loc=0., scale=1., size=3*ne)
    dn = drift_vector(a, rn, R)
    d2n = np.dot(dn, dn)
    psin = psi(a, rn, R)

    for i in range(nmc):

        chi = np.random.normal(loc=0., scale=1., size=3*ne)
        
        # Evaluate the local energy at r_n
        local_energy += e_loc(a, rn, R, Z)

        # Compute new position
        rprime = rn + dt * dn + np.sqrt(dt) * chi

        # Compute new drift
        dprime = drift_vector(a, rprime, R)
        d2prime = np.dot(dprime, dprime)

        # Evaluate Psi at the new position
        psiprime = psi(a, rprime, R)

        # Metropolis
        argexpo = np.dot((rprime - rn), (dprime + dn)) + 1./2. * dt * (d2prime - d2n)

        # Compute the ratio
        a_ratio = min(1., (psiprime / psin)**2 * np.exp(-argexpo))
        # a_ratio = (psiprime / psin)**2 * np.exp(-argexpo)
        # print('argexpo =', argexpo)
        # print('a_ratio =', a_ratio)

        # Draw an uniform random number
        v = np.random.uniform()

        # If v <= A, accept the move: set r_{n+1} = rprime
        if v <= a_ratio:
            n_accept += 1
            r_np1 = rprime
            d_np1 = dprime
            d2_np1 = d2prime
            psi_np1 = psiprime
        # else, reject the move: set r_{n+1} = rn
        else:
            r_np1 = rn
            d_np1 = dn
            d2_np1 = d2n
            psi_np1 = psin

        # Update the position
        rn = r_np1
        dn = d_np1
        d2n = d2_np1
        psin = psi_np1

    return local_energy/nmc, n_accept/nmc

# Pure Diffusion Monte Carlo
def Pure_diffusion_MC(a, ne, R, Z, nmc, dt, tau, eref):
    # a: Slater orbital exponent
    # ne: number of electrons
    # R: nucleus coordinates vector
    # Z: nucleus charge
    # nmc: number of MC steps
    # dt: time step to discretize the integral
    # tau: maximum projection time (before reseting)
    # eref: reference energy
    # a -> float
    # ne -> int
    # R -> (x1, y1, z1, x2, y2, z2, ..., xm, ym, zm)
    # Z -> (z1, z2, ..., zm)
    # nmc -> int
    # dt -> float
    # tau -> float
    # eref -> float

    # Initialize variables
    energy = 0.
    naccept = 0
    norm = 0.

    # Start with W(r0) = 1, tau0 = 0
    w = 1.
    taun = 0.

    # Generate random position & drift vector around r0 
    rn = np.random.normal(loc=0., scale=1., size=3*ne)
    dn = drift_vector(a, rn, R)
    psin = psi(a, rn, R)

    for i in range(nmc):
        # Evaluate the local energy at rn
        local_energy = e_loc(a, rn, R, Z)

        # Compute the contribution to the weight & update it
        w *= np.exp(-dt * (local_energy - eref))

        # Add up the weight for the normalization factor
        norm += w

        # and energy
        energy += w * local_energy

        # Update tau
        taun += dt

        # Reset when the long projection time has been reached
        if taun > tau:
            w = 1.0
            taun = 0.

        # Compute chi
        chi = np.random.normal(loc=0., scale=1.0, size=3*ne)

        # Update position
        rprime = rn + dt * dn + np.sqrt(dt) * chi

        # Evaluate wavefunction and drift vector at the new position
        dprime = drift_vector(a, rprime, R)
        psiprime = psi(a, rprime, R)

        # Apply the Metropolis algorithm
        prod = np.dot((dprime + dn), (rprime - rn))
        argexpo = 0.5 * (np.dot(dprime, dprime) - np.dot(dn, dn)) * dt + prod

        # Compute the ratio
        a_ratio = min(1., (psiprime / psin)**2 * np.exp(-argexpo))

        # Draw an uniform random number
        v = np.random.uniform()

        # If v <= A, accept the move: set r_{n+1} = rprime
        if v <= a_ratio:
            naccept += 1
            r_np1 = rprime
            d_np1 = dprime
            psi_np1 = psiprime
        # else, reject the move: set r_{n+1} = rn
        else:
            r_np1 = rn
            d_np1 = dn
            psi_np1 = psin

        # Update the position
        rn = r_np1
        dn = d_np1
        psin = psi_np1

    return energy/norm, naccept/nmc

def VMC(a, ne, R, Z, mc_trials, nmc, dt_VMC, dt_metro):
    # -----------------------------------------------------------------------------
    # Variational Monte Carlo algorithm
    # -----------------------------------------------------------------------------
    mc_energy_lst = []
    for i in range(mc_trials):
        mc_energy_lst.append(MC(a, ne, R, Z, nmc, dt_VMC))

    mc_energy = average(mc_energy_lst)
    mc_energy_error = error(mc_energy_lst)

    # Store the results in the QMC class
    sVMC = QMC(mc_energy, mc_energy_error, 0., 0.)
    
    # -----------------------------------------------------------------------------
    # Symmetric Metropolis algorithm
    # -----------------------------------------------------------------------------
    metropolis_sym_E = []
    metropolis_sym_ratio = []
    for i in range(mc_trials):
        x, y = Metropolis_symmetric_MC(a, ne, R, Z, nmc, dt_metro)
        metropolis_sym_E.append(x)
        metropolis_sym_ratio.append(y)

    msE = average(metropolis_sym_E)
    msE_error = error(metropolis_sym_E)
    msR = average(metropolis_sym_ratio)
    msR_error = error(metropolis_sym_ratio)
    
    # Print results
    syMe = QMC(msE, msE_error, msR, msR_error)
    
    # -----------------------------------------------------------------------------
    # Generalized Metropolis algorithm
    # -----------------------------------------------------------------------------
    metropolis_gen_E = []
    metropolis_gen_ratio = []
    for i in range(mc_trials):
        x, y = Metropolis_generalized_MC(a, ne, R, Z, nmc, dt_metro)
        metropolis_gen_E.append(x)
        metropolis_gen_ratio.append(y)

    mgE = average(metropolis_gen_E)
    mgE_error = error(metropolis_gen_E)
    mgR = average(metropolis_gen_ratio)
    mgR_error = error(metropolis_gen_ratio)
    
    # Print results
    geMe = QMC(mgE, mgE_error, mgR, mgR_error)

    return sVMC, syMe, geMe

# Print results of VMC
def output_VMC(sVMC, syMe, geMe):
    # -----------------------------------------------------------------------------
    # Monte Carlo algorithm
    section('Variational Monte Carlo')
    # -----------------------------------------------------------------------------
    
    print('E = ', float_format(sVMC.E), '+-', float_format(sVMC.s))
    
    # -----------------------------------------------------------------------------
    # Metropolis algorithm
    section('Metropolis (symmetric) MC')
    # -----------------------------------------------------------------------------
    
    print('E = ', float_format(syMe.E), '+-', float_format(syMe.s))
    print('Ratio = ', float_format(syMe.A), '+-', float_format(syMe.sA))
    
    # -----------------------------------------------------------------------------
    section('Metropolis (generalized) MC')
    # -----------------------------------------------------------------------------
    
    print('E = ', float_format(geMe.E), '+-', float_format(geMe.s))
    print('Ratio = ', float_format(geMe.A), '+-', float_format(geMe.sA))

    # =============================================================================
    # Summary
    section('Summary')
    # =============================================================================
    methods = ['VMC', 'Symmetric Metropolis', 'Generalized Metropolis']
    energies = [sVMC.E, syMe.E, geMe.E]
    errors = [sVMC.s, syMe.s, geMe.s]
    ratios = [sVMC.A, syMe.A, geMe.A]
    sigma_ratios = [sVMC.sA, syMe.sA, geMe.sA]
    
    data = pd.DataFrame({'Method': methods, 'Energy, E': energies, 'Sigma_E': errors,
                         'Ratio, A': ratios, 'Sigma_A': sigma_ratios})
    print_table("", data)

# Pure Diffusion Monte Carlo
def PDMC(a, ne, R, Z, mc_trials, nmc, dt_PDMC, tau, eref):
    # -----------------------------------------------------------------------------
    # Pure diffusion Monte Carlo
    # -----------------------------------------------------------------------------
    pure_diffusion_E = []
    pure_diffusion_ratio = []
    for i in range(mc_trials):
        x, y = Pure_diffusion_MC(a, ne, R, Z, nmc, dt_PDMC, tau, eref)
        pure_diffusion_E.append(x)
        pure_diffusion_ratio.append(y)

    pdE = average(pure_diffusion_E)
    pdE_error = error(pure_diffusion_E)
    pdR = average(pure_diffusion_ratio)
    pdR_error = error(pure_diffusion_ratio)

    # Store results
    sPDMC = QMC(pdE, pdE_error, pdR, pdR_error)

    return sPDMC

def output_PDMC(sPDMC):
    # -----------------------------------------------------------------------------
    # Pure diffusion Monte Carlo
    section('Pure Diffusion MC')
    # -----------------------------------------------------------------------------

    print('E = ', float_format(sPDMC.E), '+-', float_format(sPDMC.s))
    print('Ratio = ', float_format(sPDMC.A), '+-', float_format(sPDMC.sA))
    
    # =============================================================================
    # Summary
    section('Summary')
    # =============================================================================
    methods = ['PDMC']
    energies = [sPDMC.E]
    errors = [sPDMC.s]
    ratios = [sPDMC.A]
    sigma_ratios = [sPDMC.sA]
    
    data = pd.DataFrame({'Method': methods, 'Energy, E': energies, 'Sigma_E': errors,
                         'Ratio, A': ratios, 'Sigma_A': sigma_ratios})
    print_table("", data)
