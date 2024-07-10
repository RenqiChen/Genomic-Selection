for num in {1..5}
do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_wheat.py \
    --dataset_task_name "wheat.sterilspike1" \
    --which_k $num \
    --batch_size 116 \
    --num_workers 16 \
    --model_name gwas_transformer_wheat \
    --loss_name mse_loss \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --epoch 100 \
    --cuda
done