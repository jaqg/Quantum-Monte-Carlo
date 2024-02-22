example_xyz_files=$(shell ls examples/*.xyz)
example_xyz_outs=${example_xyz_files:.xyz=.out}

.PHONY: examples

execute:
	@python3 src/qmc.py

examples: $(example_xyz_files) examples/run_examples.sh
	@examples/run_examples.sh

clean_examples: $(example_xyz_outs)
	@rm $(example_xyz_outs)
