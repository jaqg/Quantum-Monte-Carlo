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
def MonteCarlo(a, nmc, lim):
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

    w = 1.
    taun = 0.

    # Generate random position & drift vector
    rn = np.random.normal(loc=0., scale=1., size=(3))
    dn = drift_vector(a, rn)
    psin = psi(a, rn)

    for i in range(nmc):
        # Compute the local energy
        local_energy = e_loc(a, rn)
        # Update w
        w *= np.exp(-dt * (local_energy - eref))
        # Add up normalization factor
        norm += w
        # and energy
        energy += w * local_energy
        # Update tau
        taun += dt

        # Reset when final value of tau is reached
        if taun > tau:
            w = 1.0
            taun = 0.

        # Compute chi
        chi = np.random.normal(loc=0., scale=1.0, size=(3))

        # Update position & drift vector
        rprime = rn + dt * dn + np.sqrt(dt) * chi
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
