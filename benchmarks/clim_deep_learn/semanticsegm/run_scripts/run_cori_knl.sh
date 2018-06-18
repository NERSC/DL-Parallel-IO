#!/bin/bash
#SBATCH -q debug
#SBATCH -A dasrepo
#SBATCH -C knl
#SBATCH -t 00:30:00
#SBATCH -J mascarpone_climsegment_horovod
#SBATCH -N 32
#SBATCH -o otuput_mascarpone_climsegment_horovod_run_cori_knl_20180617_nodes_32_attempt_1
#SBATCH --mail-type ALL
#SBATCH --mail-user ftc@lbl.gov

#set up python stuff

#module load gcc/6.3.0

export HDF5_USE_FILE_LOCKING=FALSE

module load tensorflow/intel-1.8.0-py27

module load h5py

#add this to library path:
modulebase=$(dirname $(module show module load tensorflow/intel-1.8.0-py27 2>&1 | grep PATH |awk '{print $3}'))
export PYTHONPATH=${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

#add this to library path:
#modulebase=$(dirname $(module show tensorflow/intel-head 2>&1 | grep PATH |awk '{print $3}'))
#export PYTHONPATH=${modulebase}/lib/python2.7/site-packages:${PYTHONPATH}

WORK=/project/projectdirs/dasrepo

#set up run directory
run_dir=${WORK}/gb2018/tiramisu/runs/cori/run_j${SLURM_JOBID}
mkdir -p ${run_dir}
cp ../tiramisu-tf/mascarpone-tiramisu-tf*.py ${run_dir}/
cp ../tiramisu-tf/tiramisu_helpers.py ${run_dir}/

#step in
cd ${run_dir}

#other directories
datadir=${WORK}/gb2018/tiramisu/segm_h5_v3_reformat

# Darshan does not function here
#module load darshan
#export LD_PRELOAD=/usr/common/software/darshan/3.1.4/lib/libdarshan.so
#export DARSHAN_LOG_PATH=$PWD
#export DARSHAN_LOGFILE=darshan.log

#run the training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u python -u mascarpone-tiramisu-tf-singlefile.py --blocks 3 3 4 4 7 7 10 --loss weighted --lr 1e-5 --datadir ${datadir} --fs global
