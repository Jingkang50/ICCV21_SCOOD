run_name=cifar100_res18_udg
id=v2

for i in 1 2 3 4 5
do
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
    python train.py \
        --config scripts_utils/${run_name}_${id}/config.yml \
        --output_dir output/${run_name}_${id}_${i}/ \
        --project_name scood \
        --run_id ${run_name}_${id}_${i} \
        --group_id ${run_name}_${id} \
        --use_wandb

    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
    python test.py \
        --config configs/test/resnet18/cifar100.yml \
        --checkpoint output/${run_name}_${id}_${i}/best.ckpt \
        --csv_path output/${run_name}_${id}_${i}/results.csv \
        --tensorboard_dir output/${run_name}_${id}_${i}/tensorboard_logs \
        --project_name scood \
        --run_id ${run_name}_${id}_${i} \
        --group_id ${run_name}_${id} \
        --use_wandb
done