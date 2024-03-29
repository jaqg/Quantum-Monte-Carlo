# Quantum Monte Carlo

This repository contains Python implementations of Monte Carlo algorithms for simulating quantum systems.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Theoretical Background](#theoretical-background)
- [Examples](#examples)
- [Contribution](#contribution)
- [License](#license)

## Project Overview
This project is aimed to develop robust and efficient QMC algorithms tailored
specifically for calculating the ground state energy of molecules.
The ground state energy is a fundamental property of a molecular system and
plays a crucial role in understanding its stability, reactivity, and other
physical properties.
By employing QMC methods, the ground state energy of molecules can be accurately 
approximated with low resources and an easy-to-use code, overcoming some of the limitations of
traditional computational chemistry techniques.

### Purpose

The primary objective of this project is to provide a humble but efficient framework
for calculating the ground state energy of molecules using Quantum Monte Carlo methods.
By employing Variational Monte Carlo (VMC) techniques, including symmetric and
general Metropolis algorithms, as well as Pure Diffusion Monte Carlo (PDMC), 
accurate estimates of the ground state energy for a variety of molecular systems
can be obtained.

### Features

- Implementation of Quantum Monte Carlo algorithms tailored for ground state energy calculations of molecules.
- Variational Monte Carlo methods with symmetric and general Metropolis algorithms.
- Pure diffusion Monte Carlo for accurate ground state energy estimation.

---
## Folder Structure

- **examples**: Contains input files for the examples H, H2, H2+, H3+, and He. Additionally, it includes the folder `reference-calculation/` with energies calculated using Gaussian software at HF/3-21G level.
- **plots**: Contains generated plots from the program, specifically from the scripts `src/make-plots.py` and `src/test-derivatives.py`.
- **src**: Contains all code and scripts.

### Source Folder (`src/`)

- **debug.py**: Contains all debugging and testing functions.
- **hamiltonian.py**: Contains functions necessary to characterize the system, such as base functions (and their derivatives), wave function (and its derivatives), potential and kinetic energy functions, and local energy.
- **IO.py**: Contains I/O and formatting process functions.
- **make_plots.py**: Generates various plots on local energy and wave function for arbitrary input data.
- **MonteCarlo.py**: Contains a class for QMC methods and all corresponding functions for the Monte Carlo method, such as statistical functions, drift vector, variational Monte Carlo (including symmetric and generalized Metropolis algorithms), and pure diffusion Monte Carlo.
- **plots.py**: Contains functions to create plots for `src/make_plots.py`.
- **test-derivatives.py**: Executes a program that finds analytical expressions for wave function derivatives. Additionally, it calculates these derivatives numerically and compares them with values obtained by functions defined in `src/hamiltonian.py`, both numerically and graphically. The plots are saved in the `plots/` folder.
- **qmc.py**: The main program to run the complete Quantum Monte Carlo simulation.

Additionally, there are two files in the `src/` folder: `mine.mplstyle` which provides custom formatting for plots generated with matplotlib, and `periodic-table.csv`, which contains necessary elements data for the program's correct functioning.

## Installation

To use this project, clone the repository to your local machine:

```bash
git clone https://github.com/jaqg/Quantum-Monte-Carlo.git
cd Quantum-Monte-Carlo
```

Make sure you have Python installed on your system. It is recommended to create a virtual environment to install the dependencies:

```bash
python -m venv venv
source venv/bin/activate # On Linux/Mac
venv\Scripts\activate # On Windows
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Once you have installed the dependencies, you can run the provided
make rules in the Makefile.

### Makefile

- **execute**: Executes the main program with input files located in the working directory.
- **examples**: Executes the main program for examples and generates output files for each of them.
- **plots**: Executes the `src/make_plots.py` program.
- **derivatives**: Executes the `src/test-derivatives.py` program.
- **clean_examples**: Deletes output files from the `examples/` folder.

## Theoretical background

The code is based on [Anthony Scemama course](https://trex-coe.github.io/qmc-lttc-2023/).

Also, some theoretical background can be found in `theory.pdf`, with the
detail derivation of the analitic expressions used in the `hamiltonian.py` 
functions.

## Examples

In the `examples/` folder you can find the XYZ input files for the following examples:
- H
- H2
- H2+
- H3+
- He

By executing 
```bash
make examples
```
the script `examples/run_examples.sh` is executed, wchich runs the main program
following the `examples/INPUT` parameters on each of the `*.xyz` file,
generating the corresponding `*.out` file.

## Contribution

Contributions are welcome! If you would like to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch for your contribution (`git checkout -b feature/new-feature`).
3. Make your changes and commit (`git commit -am 'Add new feature'`).
4. Push your branch (`git push origin feature/new-feature`).
5. Open a pull request on GitHub.

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE](LICENSE).
