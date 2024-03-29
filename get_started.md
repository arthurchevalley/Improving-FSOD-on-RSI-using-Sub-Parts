# Prerequisites


**Step 0.** Download and install mamba from the [official github]([https://docs.conda.io/en/latest/miniconda.html](https://github.com/mamba-org/mamba)).

**Step 1.** Create a mamba environment and activate it.

```shell
mamba create --name FSOD_RSI python=3.8 -y
mamba activate FSOD_RSI
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), tested with:

On GPU platforms:

```shell
mamba install pytorch==1.10.0 torchvision==0.11.1  cudatoolkit=10.2.89 -c pytorch
```

# Installation



**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install openmim
mim install mmcv-full==1.3.17
```

**Step 1.** Install MMDetection.

```shell
pip install mmdet==2.25.3
```

**Step 2.** For more complex model, TIMM and Albumentations are needed
```shell
mamba install -c conda-forge timm
pip install -U albumentations
```

**Step 3.** Multiple packages are needed and can be installed with
```shell
pip install yacs
mamba install -c conda-forge numpy=1.23.4
```
**Step 4.** Install the new models, pipeline,.. as a package
```shell
bash setup_new_models.sh
```

**Optional** Install cometML to monitor trainings etc
```shell
pip install comet-ml
```

**Pretrained Backbone**
The models are using a millionAID pre-trained backbone. The weights are downloaded from [this repo](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing). An extra processing step is needed to match the weight names.
```shell
rsp-resnet-50-ckpt_ready.pth
```