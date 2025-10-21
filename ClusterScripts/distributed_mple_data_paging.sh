#BSUB -q gsla-cpu
#BSUB -R "span[hosts=1]"
#BSUB -R rusage[mem=120000]
#BSUB -o ../logs/intermediate_calcs/outs.%J.%I.log
#BSUB -e ../logs/intermediate_calcs/errors.%J.%I.error.log
#BSUB -C 1
