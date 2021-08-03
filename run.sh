export PYTHONPATH='$PYTHONPATH:/export/nfs/xs/codes/lp-deepssl'

stage=${1:-'1'} # train stage
mt=${2:-'no'}   # whether to use mt

TIME=$(date "+%Y%m%d_%H%M%S")

DATASET='cifar10'
NOLABELS=500
GPUID=0
SPLITID=10  # 10-19
LABELED_IN_BATCH=50  # whole batchsize=100

if [ $stage = '1' ]; then  # full stage
    if [ $mt = 'mt' ]; then  # mt not use L2 norm
        cmd="python train_stage1.py --labeled-batch-size=$LABELED_IN_BATCH --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=$SPLITID --isMT=True --isL2=False --dataset=$DATASET"
        log=$(printf "%s_label%d_mt_split%d_isL2_%d_%s.log" ${DATASET} ${NOLABELS} ${SPLITID} 0 ${TIME})
    else
        cmd="python train_stage1.py --exclude-unlabeled=True --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=$SPLITID --isMT=False --isL2=True --dataset=$DATASET"
        log=$(printf "%s_label%d_split%d_isL2_%d_%s.log" ${DATASET} ${NOLABELS} ${SPLITID} 1 ${TIME})
    fi

elif [ $stage = '2' ]; then  # ss stage
    if [ $mt = 'mt' ]; then
        cmd="python train_stage2.py --labeled-batch-size=$LABELED_IN_BATCH --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=$SPLITID --isMT=True --isL2=False --dataset=$DATASET"
        log=$(printf "%s_label%d_mt_ss_split%d_isL2_%d_%s.log" ${DATASET} ${NOLABELS} ${SPLITID} 0 ${TIME})
    else
        cmd="python train_stage2.py --labeled-batch-size=$LABELED_IN_BATCH --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=$SPLITID --isMT=False --isL2=True --dataset=$DATASET"
        log=$(printf "%s_label%d_ss_split%d_isL2_%d_%s.log" ${DATASET} ${NOLABELS} ${SPLITID} 1 ${TIME})
    fi
fi

echo $cmd
echo $log

nohup $cmd > "logs/${log}" 2>&1 &
