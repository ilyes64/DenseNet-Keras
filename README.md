# DenseNet-Keras with ImageNet Pretrained Models

This is an [TensorFlow](https://www.tensorflow.org/) implementation of DenseNet with [ImageNet](http://www.image-net.org/) pretrained weights. The weights are converted from [Caffe Models](https://github.com/shicai/DenseNet-Caffe). The implementation is based on tf.keras.

To know more about how DenseNet works, please refer to the [original paper](https://arxiv.org/abs/1608.06993)

```
Densely Connected Convolutional Networks
Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten
arXiv:1608.06993
```

## Dependencies

* TensorFlow 2.0

## Pretrained DenseNet Models on ImageNet

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN)

Network|Top-1|Top-5|Tensorflow
:---:|:---:|:---:|:---:|:---:
DenseNet 121 (k=32)| 74.91| 92.19| [model (32 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc)
DenseNet 169 (k=32)| 76.09| 93.14| [model (56 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfSEc5UC1ROUFJdmM)
DenseNet 161 (k=48)| 77.64| 93.79| [model (112 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA)

## Usage

First, download the above pretrained weights to the `imagenet_models` folder.

Run `test_inference.py` for an example of how to use the pretrained model to make inference.

```
python test_inference.py
```

## Fine-tuning

Check [this](https://github.com/flyyufelix/cnn_finetune) out to see example of fine-tuning DenseNet with your own dataset.
