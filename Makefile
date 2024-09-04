a unittest:
	python -m unittest discover -s tests

profile:
	python -m cProfile -o 100_100k_sparse_matrices_2.profile profiler.py --type=big_network
	# python -m cProfile -o 100_100k_regular_matrices_2.profile profiler.py --type=big_network