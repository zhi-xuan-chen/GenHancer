export AE="/jhcnas5/chenzhixuan/checkpoints/GenHancer/ae.safetensors"
export WANDB_PROJECT="GenHancer"
export WANDB_ENTITY="zchenhi"
export CUDA_VISIBLE_DEVICES=0,5
accelerate launch --config_file "/home/chenzhixuan/Workspace/GenHancer/Continuous/train_configs/accelerate_config.yaml" --num_processes 2 /home/chenzhixuan/Workspace/GenHancer/Continuous/train_SigLIP_stage2_only.py --config "/home/chenzhixuan/Workspace/GenHancer/Continuous/train_configs/mimic_dataset_SigLIP_384_stage2_only.yaml"
