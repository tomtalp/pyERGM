a unittest:
	python -m unittest discover -s tests

profile:
	python -m cProfile -o p1_model_profile_changescore_in_out_deg.profile profiler.py --type=p1
	#python -m cProfile -o big_network_example.profile profiler.py --type=big_network