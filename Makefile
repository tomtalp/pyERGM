unittest:
	python -m unittest discover -s tests

profile:
	#python -m cProfile -o p1_fit_random_speedup.profile profiler.py --type=p1
	python -m cProfile -o p1_fit_np_random.profile profiler.py --type=p1