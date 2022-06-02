# Translation-equivariant Image Quantizer for Bi-directional Image-Text Generation

**Woncheol Shin<sup>1</sup>, Gyubok Lee<sup>1</sup>, Jiyoung Lee<sup>1</sup>, Joonseok Lee<sup>2,3</sup>, Edward Choi<sup>1</sup>** | [Paper](https://arxiv.org/abs/2112.00384)

**<sup>1</sup>KAIST, <sup>2</sup>Google Research, <sup>3</sup>Seoul National University**


## Abstract

Recently, vector-quantized image modeling has demonstrated impressive performance on generation tasks such as text-to-image generation. However, we discover that the current image quantizers do not satisfy translation equivariance in the quantized space due to aliasing, degrading performance in the downstream text-to-image generation and image-to-text generation, even in simple experimental setups. Instead of focusing on anti-aliasing, we take a direct approach to encourage translation equivariance in the quantized space. In particular, we explore a desirable property of image quantizers, called 'Translation Equivariance in the Quantized Space' and propose a simple but effective way to achieve translation equivariance by regularizing orthogonality in the codebook embedding vectors. Using this method, we improve accuracy by +22% in text-to-image generation and +26% in image-to-text generation, outperforming the VQGAN.


## Requirements

```
conda env create -f environment.yaml
conda activate bidalle
pip install horovod==0.22.1
```
If you fail to install horovod, please refer to [here](https://horovod.readthedocs.io/en/stable/).

## Download Dataset

```
bash download_mnist64x64_stage2.sh
```

## Download Image Classifier

```
bash download_classifier_ckpt.sh
```

## Training Bi-directional Image-Text Generator (Stage 2)

In ``run_train_dalle.sh``, you should specify ``--vqgan_model_path`` and ``--vqgan_config_path``.
Provide your model path pretrained from [TE-VQGAN](https://github.com/wcshin-git/TE-VQGAN).
For example, 
```
--vqgan_model_path /home/TE-VQGAN/logs/2022-04-01T07-37-39_mnist64x64_vqgan/checkpoints/last.ckpt \
--vqgan_config_path /home/TE-VQGAN/logs/2022-04-01T07-37-39_mnist64x64_vqgan/configs/2022-04-01T07-37-39-project.yaml
```
And then run the script:
```
bash run_train_dalle.sh
```

## Citation

```
@misc{shin2021translationequivariant,
      title={Translation-equivariant Image Quantizer for Bi-directional Image-Text Generation}, 
      author={Woncheol Shin and Gyubok Lee and Jiyoung Lee and Joonseok Lee and Edward Choi},
      year={2021},
      eprint={2112.00384},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

The implementation of 'TE-VQGAN' and 'Bi-directional Image-Text Generator' is based on [VQGAN](https://github.com/CompVis/taming-transformers) and [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch). 