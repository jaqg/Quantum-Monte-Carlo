from IO import print_table, section
from hydrogen import e_loc, psi
import numpy as np
import pandas as pd

# Estimated energy
def estimated_energy(ninterval, lim, a_list):
    interval = np.linspace(-lim,lim,num=ninterval)
    delta = (interval[1]-interval[0])**3
    r = np.array([0.,0.,0.])

    estimated_E = []
    for a in a_list:
        num = 0
        denom = 0
        for x in interval:
            r[0] = x
            for y in interval:
                r[1] = y
                for z in interval:
                    r[2] = z
                    wi = np.conjugate(psi(a, r)) * psi(a, r) * delta
                    num += wi * e_loc(a, r)
                    denom += wi
        estimated_E.append(num/denom)

    return estimated_E

# Variance of the local energy
def variance_eloc(ninterval, lim, a_list):
    interval = np.linspace(-lim,lim,num=ninterval)
    delta = (interval[1]-interval[0])**3
    r = np.array([0.,0.,0.])

    variance = []
    for a in a_list:
        e1 = 0.  # <E^2>
        e2 = 0.  # <E>^2
        norm = 0.
        for x in interval:
            r[0] = x
            for y in interval:
                r[1] = y
                for z in interval:
                    r[2] = z
                    wi = np.conjugate(psi(a, r)) * psi(a, r) * delta
                    norm += wi
                    e1 += wi * e_loc(a, r)**2
                    e2 += wi * e_loc(a, r)

        e1 = e1/norm
        e2 = e2/norm

        variance.append(e1 - e2**2)

    return variance

# Function to compute the average value of an array
def average(a):
    n = len(a)
    assert(n>0)

    if n == 1:
        res = a[0]
    else:
        res = sum(a)/n

    return res

# Function to compute the variance of the average
def variance(a):
    n = len(a)
    assert(n>0)

    if n == 1:
        res = 0.
    else:
        a_av = average(a)
        suma = 0.
        for ai in a:
            suma += (ai - a_av)**2
        res = suma/(len(a) - 1)

    return res

# Function to compute the error
def error(a):
    return np.sqrt(variance(a))

# Monte Carlo algorithm
def MC(a, nmc, lim):
    # Initialize variables
    local_energy = 0.
    norm = 0.
    for i in range(nmc):
        # Generate random position
        r = np.random.uniform(-lim, lim, (3))

        # Evaluate and acumulate the local energy and normalization factor
        wi = np.conjugate(psi(a,r)) * psi(a,r)
        local_energy += wi * e_loc(a, r)
        norm += wi

    return local_energy/norm

# Metropolis symmetric algorithm
def Metropolis_symmetric_MC(a, nmc, dt):
    local_energy = 0.
    n_accept = 0

    rn = np.random.uniform(-dt, dt, (3))
    psin = psi(a, rn)
    for i in range(nmc):
        
        # Evaluate the local energy at r_n
        local_energy += e_loc(a, rn)

        # Compute new position
        rprime = rn + np.random.uniform(-dt, dt, (3))

        # Evaluate Psi at the new position
        psiprime = psi(a, rprime)

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

# Function to compute the drift vecctor
def drift_vector(a, r):
    if np.dot(r,r) < 0:
        return 0
    else:
        ar_inv = -a/np.sqrt(np.dot(r,r))
        return r * ar_inv

# Metropolis generalized algorithm
def Metropolis_generalized_MC(a, nmc, dt):
    local_energy = 0.
    n_accept = 0

    rn = np.random.normal(loc=0., scale=1., size=(3))
    dn = drift_vector(a, rn)
    psin = psi(a, rn)

    for i in range(nmc):

        chi = np.random.normal(loc=0., scale=1., size=(3))
        
        # Evaluate the local energy at r_n
        local_energy += e_loc(a, rn)

        # Compute new position
        rprime = rn + dt * dn + np.sqrt(dt) * chi

        # Compute new drift
        dprime = drift_vector(a, rprime)

        # Evaluate Psi at the new position
        psiprime = psi(a, rprime)

        # Metropolis
        prod = np.dot((dprime + dn), (rprime - rn))
        argexpo = 0.5 * (np.dot(dprime, dprime) - np.dot(dn, dn)) * dt + prod

        # Compute the ratio
        a_ratio = min(1., (psiprime / psin)**2 * np.exp(-argexpo))

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

# Pure Diffusion Monte Carlo
def Pure_diffusion_MC(a, nmc, dt, tau, eref):
    # Initialize variables
    energy = 0.
    naccept = 0
    norm = 0.

    # Start with W(r0) = 1, tau0 = 0
    w = 1.
    taun = 0.

    # Generate random position & drift vector
    rn = np.random.normal(loc=0., scale=1., size=(3))
    dn = drift_vector(a, rn)
    psin = psi(a, rn)

    for i in range(nmc):
        # Evaluate the local energy at rn
        local_energy = e_loc(a, rn)

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
        chi = np.random.normal(loc=0., scale=1.0, size=(3))

        # Update position
        rprime = rn + dt * dn + np.sqrt(dt) * chi

        # Evaluate wavefunction and drift vector at the new position
        dprime = drift_vector(a, rprime)
        psiprime = psi(a, rprime)

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

# Variational Monte Carlo
def VMC(a, mc_trials, nmc, lim, dt):
    # -----------------------------------------------------------------------------
    # Monte Carlo algorithm
    section('Variational Monte Carlo')
    # -----------------------------------------------------------------------------
    
    mc_energy = []
    for i in range(mc_trials):
        mc_energy.append(MC(a, nmc, lim))
    
    # Print results
    variational_mc_E = average(mc_energy)
    variational_mc_sigma = error(mc_energy)
    print('E = ', variational_mc_E, '+-', variational_mc_sigma)
    
    # -----------------------------------------------------------------------------
    # Metropolis algorithm
    section('Metropolis (symmetric) MC')
    # -----------------------------------------------------------------------------
    dt = 1.
    
    metropolis_sym_E = []
    metropolis_sym_ratio = []
    for i in range(mc_trials):
        x, y = Metropolis_symmetric_MC(a, nmc, dt)
        metropolis_sym_E.append(x)
        metropolis_sym_ratio.append(y)
    
    # Print results
    sym_metropolis_E = average(metropolis_sym_E)
    sym_metropolis_sigma_E = error(metropolis_sym_E)
    sym_metropolis_ratio = average(metropolis_sym_ratio)
    sym_metropolis_sigma_ratio = error(metropolis_sym_ratio)
    
    print('E = ', sym_metropolis_E, '+-', sym_metropolis_sigma_E)
    print('Ratio = ', sym_metropolis_ratio, '+-', sym_metropolis_sigma_ratio)
    
    section('Metropolis (generalized) MC')
    
    metropolis_gen_E = []
    metropolis_gen_ratio = []
    for i in range(mc_trials):
        x, y = Metropolis_generalized_MC(a, nmc, dt)
        metropolis_gen_E.append(x)
        metropolis_gen_ratio.append(y)
    
    # Print results
    gen_metropolis_E = average(metropolis_gen_E)
    gen_metropolis_sigma_E = error(metropolis_gen_E)
    gen_metropolis_ratio = average(metropolis_gen_ratio)
    gen_metropolis_sigma_ratio = error(metropolis_gen_ratio)
    
    print('E = ', gen_metropolis_E, '+-', gen_metropolis_sigma_E)
    print('Ratio = ', gen_metropolis_ratio, '+-', gen_metropolis_sigma_ratio)

    # =============================================================================
    # Summary
    section('Summary')
    # =============================================================================
    methods = ['VMC', 'Symmetric Metropolis', 'Generalized Metropolis']
    
    energies = [variational_mc_E,
                sym_metropolis_E,
                gen_metropolis_E]    

    errors = [variational_mc_sigma,
              sym_metropolis_sigma_E,
              gen_metropolis_sigma_E]
    
    ratios = [0.,
              sym_metropolis_ratio,
              gen_metropolis_ratio]
    
    sigma_ratios = [0.,
                    sym_metropolis_sigma_ratio,
                    gen_metropolis_sigma_ratio]
    
    data = pd.DataFrame({'Method': methods, 'Energy, E': energies, 'Sigma_E': errors,
                         'Ratio, A': ratios, 'Sigma_A': sigma_ratios})
    print_table("", data)

# Pure Diffusion Monte Carlo
def PDMC(a, mc_trials, nmc, dt, tau, eref):
    # -----------------------------------------------------------------------------
    # Pure diffusion Monte Carlo
    section('Pure Diffusion MC')
    # -----------------------------------------------------------------------------

    pure_diffusion_E = []
    pure_diffusion_ratio = []

    for i in range(mc_trials):
        x, y = Pure_diffusion_MC(a, nmc, dt, tau, eref)
        pure_diffusion_E.append(x)
        pure_diffusion_ratio.append(y)
    
    # Print results
    pd_E = average(pure_diffusion_E)
    pd_sigma_E = error(pure_diffusion_E)
    pd_ratio = average(pure_diffusion_ratio)
    pd_sigma_ratio = error(pure_diffusion_ratio)
    
    print('E = ', pd_E, '+-', pd_sigma_E)
    print('Ratio = ', pd_ratio, '+-', pd_sigma_ratio)
    
    # =============================================================================
    # Summary
    section('Summary')
    # =============================================================================
    methods = ['PDMC']
    
    energies = [pd_E]
    
    errors = [pd_sigma_E]
    
    ratios = [pd_ratio]
    
    sigma_ratios = [pd_sigma_ratio]
    
    data = pd.DataFrame({'Method': methods, 'Energy, E': energies, 'Sigma_E': errors,
                         'Ratio, A': ratios, 'Sigma_A': sigma_ratios})
    print_table("", data)

