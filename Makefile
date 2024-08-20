a unittest:
	python -m unittest discover -s tests

profile:
	python -m cProfile -o big_network_1_sample_statistics_no_loop.profile profiler.py --type=big_network