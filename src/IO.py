import numpy as np
import pandas as pd
from tabulate import tabulate
import logging
import os

def read_xyz(xyz_filename):
    # Read the title & charge
    xyz_file = open(xyz_filename, "r")
    title = str(xyz_file.readline())
    charge = int(xyz_file.readline())
    xyz_file.close()

    # Read the atoms & xyz coordinates
    nxyz = pd.read_csv(xyz_filename, skiprows=2, index_col=False, 
                       delimiter=r"\s+", names=['atom', 'x', 'y', 'z'])

    # Change coordinates from angstrom to atomic units
    angstrom_to_au = 1.8897259886
    nxyz[['x','y','z']] *= angstrom_to_au

    # Add a new column with the number of electrons of each atom;
    # open file with periodic table information
    pt_file = os.path.dirname(__file__)+'/{}'.format('periodic-table.csv')
    periodic_table = pd.read_csv(pt_file)

    # For each atom on the list, find it in the periodic table and add its
    # atomic number and number of electrons in new columns
    electrons = []
    atomic_numbers = []
    for i in range(len(nxyz)):
        atom = nxyz['atom'][i]
        for j in range(len(periodic_table)):
            if periodic_table['Symbol'][j] == atom:
                electrons.append(periodic_table['NumberofElectrons'][j])
                atomic_numbers.append(periodic_table['AtomicNumber'][j])

    nxyz['nelectrons'] = pd.Series(electrons)
    nxyz['atomic_number'] = pd.Series(atomic_numbers)

    return title, charge, nxyz

# Function to extract coordinates, number of electrons and Z from xyz dataframe
def extract_R_ne(charge, nxyz):
    # Sum the number of electrons and apply the charge
    ne = sum(nxyz['nelectrons']) - charge

    # Store the atom coordinates and atomic numbers in single vectors:
    # R = [x1, y1, z1, x2, y2, z2, ..., x_m, y_m, z_m]
    # Z = [Z1, Z2, ..., Z_m]
    R = []
    Z = []
    for index, row in nxyz.iterrows():
        R.extend([row['x'], row['y'], row['z']])
        Z.extend([row['atomic_number']])

    return ne, R, Z

# Function to read the input file
def read_input(input_filename):
    # Open input_file
    input_file = open(input_filename, "r") 

    # Read type of calculation
    input_file.readline()  # skip comment line
    calculation_type = input_file.readline()
    calculation_type = calculation_type.strip()
    if calculation_type == 'VMC' or calculation_type == 'vmc':
        calculation_type = 'VMC'
    elif calculation_type == 'PDMC' or calculation_type == 'pdmc':
        calculation_type = 'PDMC'
    else:
        print("Error: unknown calculation type")
        exit(1)

    # Read exponent factor, a
    input_file.readline()
    a = float(input_file.readline())

    # Read number of steps for the MC calculation, nmc
    input_file.readline()
    nmc = int(input_file.readline())

    # Read number of MC calculations, mc_trials
    input_file.readline()
    mc_trials = int(input_file.readline())

    # Read value of dt for VMC
    input_file.readline()
    dt_VMC = float(input_file.readline())

    # Read value of dt for metropolis
    input_file.readline()
    dt_metro = float(input_file.readline())

    # Read value of dt for PDMC
    input_file.readline()
    dt_PDMC = float(input_file.readline())

    # Read value of tau
    input_file.readline()
    tau = float(input_file.readline())

    # Read value of eref
    input_file.readline()
    eref = float(input_file.readline())

    # Close input_file
    input_file.close() 
    return calculation_type, a, nmc, mc_trials, dt_VMC, dt_metro, dt_PDMC, tau, eref

def print_input(title, a, charge, nxyz, ne, R, Z):
    print('title:', title)
    print('Slater orbital exponent, a:', a)
    print('\nxyz (in a.u.):\n', nxyz[['atom', 'x', 'y', 'z', 'atomic_number', 'nelectrons']])
    print('\nNumber of electrons:', ne + charge)
    print('Charge:', charge)
    print('Effective number of electrons (considering charge):', ne)

def print_MC_data(mc_trials, nmc, dt_VMC, dt_metro, dt_PDMC, tau, eref):
    print('Number of MC steps:', nmc)
    print('Number of MC simulations:', mc_trials)
    print('dt(VMC):', dt_VMC)
    print('dt(metropolis):', dt_metro)
    print('dt(PDMC):', dt_PDMC)
    print('tau (for PDMC):', tau)
    print('Eref (for PDMC):', eref)


# -----------------------------------------------------------------------------
# Formatting functions
# -----------------------------------------------------------------------------
def float_format(x):
    return '{:0.6f}'.format(x)

def section(string):
    print("\n" + "+" + "-"*78 + "+")
    print("|" + string.center(78) + "|")
    print("+" + "-"*78 + "+" + "\n")

def title(string):
    print("\n" + "+" + "="*78 + "+")
    print("|" + string.center(78) + "|")
    print("+" + "="*78 + "+" + "\n")

def end():
    print('Bye! :)' + ' '*44 + '~José Antonio Quiñonero Gris')
    print("+" + "="*78 + "+")

# Function to print table
def print_table(title, data):
    print(title)
    print(tabulate(data, floatfmt=".4f", showindex=False, headers='keys',
                   tablefmt="grid"), end='\n\n')

# Function to add logging level
def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)
