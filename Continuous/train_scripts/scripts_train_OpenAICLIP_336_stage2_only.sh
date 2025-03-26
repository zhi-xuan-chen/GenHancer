export AE="/group/40034/jasonsjma/models_hf/FLUX.1-dev/ae.safetensors"

accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_stage2_only.py --config "train_configs/test_OpenAICLIP_336_stage2_only.yaml"
