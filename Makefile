example_xyz_files=$(shell ls examples/*.xyz)
example_xyz_outs=${example_xyz_files:.xyz=.out}

.PHONY: examples plots derivatives

execute:
	@python3 'src/qmc.py'

examples: $(example_xyz_files) examples/run_examples.sh
	@examples/run_examples.sh

plots: src/make_plots.py
	@python3 'src/make_plots.py'

derivatives:
	@python3 'src/test-derivatives.py'

clean_examples: $(example_xyz_outs)
	@rm $(example_xyz_outs)
