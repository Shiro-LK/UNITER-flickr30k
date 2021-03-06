# UNITER-flick30k
UNITER Finetuned on flickr30k dataset using the official repository of UNITER : https://github.com/ChenRocks/UNITER

One gpu has been used for the finetuning, the hyper-parameters chosen where therefore not optimal.
The results obtained are : 
|           | R1         | 
| ------------- |:-------------:| 
| Text      | 81 | 
| Image      |  66     | 


A naive deployment has been implemented in this repo with Flask. The finetuned weights is in the folder UI.


## Requirement
- ubuntu 18.04
- nvidia-driver installed (tested with version 450) : https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation
- cuda 10.1
- docker >= 19.03
- nvidia docker : https://github.com/NVIDIA/nvidia-docker#quickstart

Step to launch the deployment:


## Launch Docker
clone repo :

> git clone https://github.com/Shiro-LK/UNITER-flickr30k


> cd UNITER-flickr30k

pretrained weights : 
> https://drive.google.com/file/d/1s2OFoH8wT8EK_XZuWbsnNVUTezJFwQ-K/view?usp=sharing


and put the weights inside : UNITER-flickr30K/UI





build :
> docker build -t uniter-deployment .

run the container :
> docker run -it --gpus 0 -p 5000:5000 -v path/to/UNITER-flickr30k/UI:/UI uniter-deployment bash

inside the container :
> cd /UI

> export LC_ALL=C.UTF-8

> flask run --host=0.0.0.0

then write the text and select the image in the UI you want to compare. (The image need to be inside the folder UI/app/static/images)

![image](images/image0.jpeg)


