#!/bin/bash

(
	cd "examples" || exit

	xyz_files=$(find . -name "*.xyz")

	for file in $xyz_files
	do
		output_file=${file%.xyz}.out
		cp "$file" "XYZ"
		echo "Running $(basename "$file" .xyz)"
		python3 ../src/qmc.py > "$output_file"
	done
	rm XYZ
	echo ""
	echo "Done! Output files generated. Bye!"
)
