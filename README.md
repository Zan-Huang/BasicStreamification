## Zan's Modifications of Dense Predictive Coding 

Note by Zan: Link: https://github.com/tcapelle/torch_moving_mnist.git
is used to generate the MNIST downstream dataset. Thank you.

This repository contains the implementation of Dense Predictive Coding (DPC). 

Links: [[Arxiv](https://arxiv.org/abs/1909.04656)] [[Video](https://youtu.be/43KIHUvHjB0)] [[Project page](http://www.robots.ox.ac.uk/~vgg/research/DPC/dpc.html)]

![arch](asset/arch.png)


### Installation

The implementation should work with python >= 3.6, pytorch >= 0.4, torchvision >= 0.2.2. 

The repo also requires cv2 (`conda install -c menpo opencv`), tensorboardX >= 1.7 (`pip install tensorboardX`), joblib, tqdm, ipdb.

### Prepare data

Follow the instructions [here](process_data/).

### Self-supervised training (DPC)

Change directory `cd DPC/dpc/`

* example: train DPC-RNN using 2 GPUs, with 3D-ResNet18 backbone, on Kinetics400 dataset with 128x128 resolution, for 300 epochs

  ------------------------------------------------------------------------------------------
  BASIC BASIC BASIC BASIC TRAINING
  '''
  python main.py --gpu 0,1,2,3 --net dpc_basic --dataset mnist --batch_size 128 --img_dim 64 --epochs 100
  '''
  ------------------------------------------------------------------------------------------

  Basic two stream pretraining:
  '''
  python main.py --gpu 0,1,2,3,4,5,6,7 --net basictwostream --dataset mnist --batch_size 128 --img_dim 64 --epochs 100
  '''

  ------------------------------------------------------------------------------------------

* example: train DPC-RNN using 4 GPUs, with 3D-ResNet34 backbone, on Kinetics400 dataset with 224x224 resolution, for 150 epochs
  ```
  python main.py --gpu 0,1,2,3 --net resnet34 --dataset k400 --batch_size 44 --img_dim 224 --epochs 150
  ```

### Evaluation: supervised action classification

Change directory `cd DPC/eval/`

* example: finetune pretrained DPC weights (replace `{model.pth.tar}` with pretrained DPC model)

  SINGLE STREAM
  --------------------------------------------------------------------------------------------
  This is for MNIST single stream

  ```
  python test.py --gpu 2,3,4,5 --net dpc_basic --dataset digits --batch_size 128 --img_dim 64 --pretrain {model_best_epoch100.pth.tar} --train_what ft --epochs 100
  ```
  This is for Motion single stream.

  ```
    python test.py --gpu 0,1,6,7 --net dpc_basic --dataset motion --batch_size 128 --img_dim 64 --pretrain {model_best_epoch100.pth.tar} --train_what ft --epochs 100
  ```

  --------------------------------------------------------------------------------------------


  This is for MNIST dual stream basic.
  ```
  python test.py --gpu 2,3,4,5 --net basictwostream --dataset digits --batch_size 128 --img_dim 64 --pretrain {modelepoch.pth.tar} --train_what ft --epochs 100
  ```

  This is for Motion dual stream basic.
  ```
  python test.py --gpu 0,1,6,7 --net basictwostream --dataset motion --batch_size 128 --img_dim 64 --pretrain {modelepoch.pth.tar} --train_what ft --epochs 100
  ```
-------------------------------------

  This is for MP Dual stream.
   ```
  python test.py --gpu 2,3,4,5 --net mptwostream --dataset motion --batch_size 256 --img_dim 128 --pretrain {model_best_epoch243.pth.tar} --train_what ft --epochs 10
  ``` 


* example (continued): test the finetuned model (replace `{finetune_model.pth.tar}` with finetuned classifier model)
  ```
  python test.py --gpu 0,1 --net resnet18 --dataset ucf101 --batch_size 128 --img_dim 128 --test {finetune_model.pth.tar}
  ```



