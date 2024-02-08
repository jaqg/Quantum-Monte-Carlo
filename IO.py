from tabulate import tabulate

def read_input(filename):
    # Open file
    file = open(filename, "r") 

    # Read type of calculation
    file.readline()
    calculation_type = file.readline()
    calculation_type = calculation_type.strip()
    if calculation_type == 'VMC' or calculation_type == 'vmc':
        calculation_type = 'VMC'
    elif calculation_type == 'PDMC' or calculation_type == 'pdmc':
        calculation_type = 'PDMC'
    else:
        print("Error: unknown calculation type")
        exit(1)

    # Read exponent factor, a
    file.readline()
    a = float(file.readline())

    # Read number of steps for the MC calculation, nmc
    file.readline()
    nmc = int(file.readline())

    # Read number of MC calculations, mc_trials
    file.readline()
    mc_trials = int(file.readline())

    # Read lim
    file.readline()
    lim = float(file.readline())

    # Read value of dt
    file.readline()
    dt = float(file.readline())

    # Read value of tau
    file.readline()
    tau = float(file.readline())

    # Read value of eref
    file.readline()
    eref = float(file.readline())

    # Close file
    file.close() 
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
