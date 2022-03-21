run_name=cifar100_wrn_odin
id=v1

for i in 1 2 3 4 5
do
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
    python test.py \
        --config configs/test/wrn/cifar100_odin.yml \
        --checkpoint output/cifar100_wrn_bl_${id}_${i}/best.ckpt \
        --csv_path output/cifar100_wrn_bl_${id}_${i}/results.csv \
        --tensorboard_dir output/${run_name}_${id}_${i}/tensorboard_logs \
        --project_name scood \
        --run_id ${run_name}_${id}_${i} \
        --group_id ${run_name}_${id} \
        --use_wandb
done