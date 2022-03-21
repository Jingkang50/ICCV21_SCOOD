run_name=cifar100_wrn_udg
id=v1

for i in 1 2 3 4 5
do
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
    python train.py \
        --config scripts_utils/${run_name}_${id}/config.yml \
        --test_config configs/test/wrn/cifar100.yml \
        --output_dir output/${run_name}_${id}_${i}/ \
        --seed ${i} \
        --project_name scood \
        --run_id ${run_name}_${id}_${i} \
        --group_id ${run_name}_${id} \
        --use_wandb
done