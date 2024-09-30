#BSUB -q short
#BSUB -R "span[hosts=1]"
#BSUB -R rusage[mem=1000]
#BSUB -o ../logs/intermediate_calcs/outs.%J.%I.log
#BSUB -e ../logs/intermediate_calcs/errors.%J.%I.error.log
#BSUB -C 1