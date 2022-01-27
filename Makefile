help:
	cat Makefile

run:
	pip install -Uqe .
	cd tests && python nbflow.py --package-suffixes ".ipynb" run

run-show-input:
	pip install -Uqe .
	cd tests && python nbflow.py --package-suffixes ".ipynb" run --exclude_nb_input False

render: run
	cd tests && python nbflow.py card view end && cd ..

render-show-input: run-show-input
	cd tests && python nbflow.py card view end && cd ..

test-show-input: run-show-input
	cd tests && ./test.sh "-show-input"

test-hide-input: run
	cd tests &&  ./test.sh ""

test: test-show-input test-hide-input

example-dl: .FORCE
	cd examples/deep_learning && python dl_flow.py --package-suffixes=".ipynb"  run
	cd examples/deep_learning &&  python dl_flow.py card view nb_auto
	cd examples/deep_learning &&  python dl_flow.py card view nb_manual

example-rf: .FORCE
	cd examples/random_forest && python flow.py --package-suffixes=".ipynb"  run
	cd examples/random_forest && python flow.py card view evaluate

.FORCE:

release:
	python setup.py sdist bdist_wheel
	twine upload dist/*
