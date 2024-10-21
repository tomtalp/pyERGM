a unittest:
	python -m unittest discover -s tests

profile:
	python -m cProfile -o arxiv_network.profile profiler.py --type=arxiv_network