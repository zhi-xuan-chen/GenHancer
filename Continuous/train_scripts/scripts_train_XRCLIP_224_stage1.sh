export AE="/jhcnas5/chenzhixuan/checkpoints/GenHancer/ae.safetensors"
export WANDB_PROJECT="GenHancer"
export WANDB_ENTITY="zchenhi"
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file "/home/chenzhixuan/Workspace/GenHancer/Continuous/train_configs/accelerate_config.yaml" --num_processes 4 /home/chenzhixuan/Workspace/GenHancer/Continuous/train_XRCLIP_stage1.py --config "/home/chenzhixuan/Workspace/GenHancer/Continuous/train_configs/combined_dataset_XRCLIP_224_stage1.yaml"
