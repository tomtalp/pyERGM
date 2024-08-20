unittest:
	python -m unittest discover -s tests

profile:
	python -m cProfile -o p1_model_profile.profile profiler.py --type=p1
	#python -m cProfile -o big_network_example.profile profiler.py --type=big_network