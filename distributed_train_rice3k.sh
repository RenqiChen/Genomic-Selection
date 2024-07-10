for num in {1..5}
do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_rice3k.py \
    --dataset_task_name "rice3k.pth" \
    --dataset_task_name_1 "pth" \
    --dataset_task_name_2 "rice3k.pth" \
    --which_k $num \
    --batch_size 128 \
    --num_workers 16 \
    --model_name gwas_transformer_rice3k \
    --loss_name ce_loss \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --epoch 100 \
    --cuda
done