unittest:
	python -m unittest discover -s tests

profile:
	# python -m cProfile -o arxiv_network_num_edges_triangles.profile profiler.py --type=arxiv_network
	python -m cProfile -o test_with_oren_no_delete mcmc_chain_tests.py

pypi_deploy:
	python setup.py sdist bdist_wheel
	twine upload dist/*

