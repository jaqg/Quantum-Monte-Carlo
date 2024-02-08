import numpy as np
import pandas as pd
from tabulate import tabulate

def read_xyz(xyz_filename):
    # Read the charge
    xyz_file = open(xyz_filename, "r")
    charge = int(xyz_file.readline())
    xyz_file.close()

    # Read the atoms & xyz coordinates
    nxyz = pd.read_csv(xyz_filename, skiprows=1, index_col=False, 
                       delimiter=r"\s+", names=['atom', 'x', 'y', 'z'])

    # Add a new column with the number of electrons of each atom;
    # open file with periodic table information
    periodic_table = pd.read_csv('periodic-table.csv')

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

    return nxyz

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

# -----------------------------------------------------------------------------
# Formatting functions
# -----------------------------------------------------------------------------
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
