help:
	cat Makefile

render:
	pip install -Uqe . && cd tests && python nbflow.py run && python nbflow.py card view end && cd ..

render-show-input::
	pip install -Uqe . && cd tests && python nbflow.py run --exclude_nb_input False && python nbflow.py card view end && cd ..