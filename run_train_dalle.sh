export CUDA_VISIBLE_DEVICES=0,1

horovodrun -np 2 python -u train_dalle.py \
    --vqgan_model_path [your_model_ckpt_path] \
    --vqgan_config_path [your_project_yaml_path] \
    --seed 42 \
    --img_classifier_path classifier_ckpt/MNIST_Resnet18.ckpt --parser MNIST \
    --image_text_folder datasets/MNIST64x64_Stage2 --valtest_name mnist_test_unseen --val_use_ratio 0.25 \
    --bpe_path BPE/mnist_bpe.json --saving_interval 2 \
    --dim 256 --text_seq_len 50 --depth 8 --heads 8 --dim_head 64 --loss_weight 7 --attn_types full,axial_row,axial_col,conv_like --pe_type fixed \
    --task alternate --epochs 20 --batch_size 100 --learning_rate 1e-3 --clip_grad_norm 1.0 --lr_decay --num_workers 8 \
    --distributed_backend horovod