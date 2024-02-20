import numpy as np
import pandas as pd
from tabulate import tabulate
import logging

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
    periodic_table = pd.read_csv('src/periodic-table.csv')

    # For each atom on the list, find it in the periodic table and add its
    # number of electrons in a new column
    electrons = []
    for i in range(len(nxyz)):
        atom = nxyz['atom'][i]
        for j in range(len(periodic_table)):
            if periodic_table['Symbol'][j] == atom:
                electrons.append(periodic_table['NumberofElectrons'][j])

    # apply the charge to the first atom on the list
    electrons[0] -= charge

    nxyz['nelectrons'] = pd.Series(electrons)

    return title, charge, nxyz

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

    # Read lim
    input_file.readline()
    lim = float(input_file.readline())

    # Read value of dt
    input_file.readline()
    dt = float(input_file.readline())

    # Read value of tau
    input_file.readline()
    tau = float(input_file.readline())

    # Read value of eref
    input_file.readline()
    eref = float(input_file.readline())

    # Close input_file
    input_file.close() 
    return calculation_type, a, nmc, mc_trials, lim, dt, tau, eref

def print_input(title, charge, nxyz):
    print('title:', title)
    print('Number of electrons:', sum(nxyz['nelectrons']) + charge)
    print('Charge:', charge)
    print('Total number of electrons:', sum(nxyz['nelectrons']))
    print('xyz (in a.u.):\n', nxyz[['atom', 'x', 'y', 'z']])

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
