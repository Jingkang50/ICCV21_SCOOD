run_name=cifar100_wrn_bl
id=v2

for i in 6
do
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
    python train.py \
        --config scripts_utils/${run_name}_${id}/config.yml \
        --test_config scripts_utils/${run_name}_${id}/test_config.yml \
        --output_dir output/${run_name}_${id}_${i}/ \
        --seed ${i} \
        --project_name scood \
        --run_id ${run_name}_${id}_${i} \
        --group_id ${run_name}_${id} \
        --use_wandb
done