from tabulate import tabulate

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
