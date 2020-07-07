
#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=224gb:np100s=1:os7=True:pcmem=8gb
### Specify a name for the job
#PBS -N 1_contextual
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=1120:00:00
### Walltime is how long your job will run
#PBS -l walltime=40:00:00
#PBS -e /xdisk/vikasy/TechQA_dataset/error/
#PBS -o /xdisk/vikasy/TechQA_dataset/output/

#####module load cuda80/neuralnet/6/6.0
#####module load cuda80/toolkit/8.0.61
module load singularity/3/3.2.1

cd /xdisk/vikasy/transformers/


export GLUE_DIR=/xdisk/vikasy/TechQA_dataset/TechQA_Gold_document_Regression_experiment_2
export OUT_DIR=/xdisk/vikasy/TechQA_REGRESSION_models/TechQA_Gold_document_Regression_experiment_2/TechQA_Gold_document_Regression_experiment_2_1_4Epochs_128_RoBERTa_base


singularity -s run --nv /xdisk/vikasy/hpc-ml_centos7-python37.sif python3.7 examples/run_glue.py     --model_type roberta     --model_name_or_path roberta-base     --do_train      --do_eval       --task_name=sts-b         --data_dir=${GLUE_DIR}      --output_dir=${OUT_DIR}       --max_seq_length=512       --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --gradient_accumulation_steps=8     --num_train_epochs=4.0     --model_name=roberta-base       --overwrite_output_dir       --overwrite_cache 

