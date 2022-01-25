help:
	cat Makefile

run:
	pip install -Uqe .
	cd tests && python nbflow.py run

run-show-input:
	pip install -Uqe .
	cd tests && python nbflow.py run --exclude_nb_input False

render: run
	cd tests && python nbflow.py card view end && cd ..

render-show-input: run-show-input
	cd tests && python nbflow.py card view end && cd ..

test-show-input: run-show-input
	cd tests && ./test.sh "-show-input"

test-hide-input: run
	cd tests &&  ./test.sh ""

test: test-show-input test-hide-input
