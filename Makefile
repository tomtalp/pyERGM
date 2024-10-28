unittest:
	python -m unittest discover -s tests

profile:
	python -m cProfile -o arxiv_network_num_edges_triangles.profile profiler.py --type=arxiv_network

pypi_deploy:
	python setup.py sdist bdist_wheel
	twine upload dist/*

