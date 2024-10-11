#BSUB -q schneidman
#BSUB -R "span[hosts=1]"
#BSUB -R rusage[mem=4000]
#BSUB -o ../logs/outs.%J.%I.log
#BSUB -e ../logs/errors.%J.%I.error.log
#BSUB -C 1
python ./distributed_mple_fit.py