# Classification_Pytorch
Various Classification Models using Pytorch 

## Support Model
VGGNet, ResNet, MobileNet V2, ResNeXt..

## Requirements
Python 3.6 or later, torch >= 1.5

## To Train
This model is adopt with Cifar-10 dataset. You need to tuning the model for your dataset.

```bash
$ python train.py -m 'ResNeXt50' -s 1 -b 64 -n 2 -e 300 -f 3 -g 5 -sh 0 -l 0.1 -op 0 -card 32 -bw 4
```

## To Test
```bash
$ python test.py
```