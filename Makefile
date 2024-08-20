a unittest:
	python -m unittest discover -s tests

profile:
	python -m cProfile -o big_network_njit_but_not_edge_flips.profile profiler.py --type=big_network