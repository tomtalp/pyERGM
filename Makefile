unittest:
	python -m unittest discover -s tests

profile:
	python -m cProfile -o profiling/global_zeros_3 profiler.py

pypi_deploy:
	python setup.py sdist bdist_wheel
	twine upload dist/*

benchmarking_to_cluster:
	cp ClusterScripts/benchmarking.sh /Volumes/tomta/pyERGM/ClusterScripts/
	cp cluster_benchmarks.py /Volumes/tomta/pyERGM/
	cp Makefile /Volumes/tomta/pyERGM/
	cp -r pyERGM/datasets.py /Volumes/tomta/pyERGM/

run_benchmark:
	bsub < ClusterScripts/benchmarking.sh -J benchmarking[1-2000] 
	bsub < ClusterScripts/benchmarking.sh -J benchmarking[2001-4000] 
	bsub < ClusterScripts/benchmarking.sh -J benchmarking[4001-6000]
	bsub < ClusterScripts/benchmarking.sh -J benchmarking[6001-6144] 

