#!/bin/sh

# Script to extract the energies from the Gaussian output file

# Function to print a separator
separator() {
	echo "------------------------"
}

# Filename of the output file
output_file='examples.log'

# Extract the molecule name
molecule_name=($(grep '%chk=' $output_file | cut -d'=' -f2 | cut -d'.' -f1))

# Extract the HF energies
energies=($(grep 'SCF Done:' $output_file | cut -d'=' -f2 | cut -d'A' -f1))

# Print the results
separator
printf '%s\t%s\n' "System" "Energy (Hartree)"
separator
for i in $(seq 0 1 $(( ${#molecule_name[@]} - 1)))
do
	printf "%s\t%.6f\n" "${molecule_name[$i]}" "${energies[$i]}"
done
separator
