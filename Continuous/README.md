# GenHancer with Continuous Generative Models

[toc]

## Introduction

* For the continuous denoiser, we employ rectified flow and choose a FLUX-like architectures, but with less DiT blocks, *i.e.,* 2 double block and 4 single blocks.
* The rectified flow model is built upon the pre-trained latent space of VAE.



## Prepare

### Installation

1. Clone this repository and navigate to Continuous folder

   ```shell
   git clone git@github.com:mashijie1028/GenHancer.git
   cd Continuous
   ```

2. Install packages:

   ```
   pip install -r requirements.txt
   ```

### Dataset

* Download CC3M dataset. Please refer to [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) for more details.
* Specify the dataset path in  `img_dir` of  `train_configs/*.yaml`.

### VAE Model

* Download the VAE model 
* Specify the `AE` environment variable in shell scripts.



## Training

### Stage-1

In Stage-1, we only train the project and denoiser while freezing the CLIP model.

```shell
# for OpenAICLIP@224
bash train_scripts/scripts_train_OpenAICLIP_224_stage1.sh

# for OpenAICLIP@336
bash train_scripts/scripts_train_OpenAICLIP_336_stage1.sh

# for SigLIP@384
bash train_scripts/scripts_train_SigLIP_384_stage1.sh
```

### Stage-2

In Stage-1, we mainly tune the CLIP model, while we empirically found that whether to freeze the projector and denoiser has little impact on the results.

Please specify `load_dir` as the saved directory of Stage-1 in yaml files.

If you tune only CLIP ViT:

```shell
# for OpenAICLIP@224
bash train_scripts/scripts_train_OpenAICLIP_224_stage2_only.sh

# for OpenAICLIP@336
bash train_scripts/scripts_train_OpenAICLIP_336_stage2_only.sh

# for OpenAICLIP@224
bash train_scripts/scripts_train_SigLIP_384_stage2_only.sh
```

If you want to tune all the components:

```shell
# for OpenAICLIP@224
bash train_scripts/scripts_train_OpenAICLIP_224_stage2_all.sh

# for OpenAICLIP@336
bash train_scripts/scripts_train_OpenAICLIP_336_stage2_all.sh

# for OpenAICLIP@224
bash train_scripts/scripts_train_SigLIP_384_stage2_all.sh
```
