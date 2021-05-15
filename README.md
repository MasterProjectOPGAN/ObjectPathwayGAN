# ObjectPathwayGAN
Synthesizing high-quality photo-realistic images is one of the challenging problems in computer vision and has a plethora of practical applications. Of many techniques that are available to solve this, Generative Adversarial Networks (GANs) have achieved greater success in recent times. 
<br />

Similarly, generating images conditioned on textual description has many applications, and to list a few – a quick visual summary for a text paragraph to enhance the learning experience for students and real-time image generation in sports for a commentary text. 
<br />

Object Pathway GAN (OP-GAN) is the baseline model, which will generate photo-realistic images consisting of multiple objects described in the text description. 
<br />
For example, if the textual description is, "two skiers are posing on a snowy slope", the objects in the image are "two skiers" and the "snowy slope". The generator network takes labels and other images' metadata like bounding box location as input to generate fine-grained images. Whereas the discriminator network feeds on the real-image and fake image to classify them into the right bucket. 
<br/ >

The project focuses on improvising the discriminator network from the original implementation. This eventually helps the generator network's learning rate adjustment based on the discriminator output. Finally, the trained model is evaluated using Semantic Object Accuracy (SOA) which is the same as the original implementation. This new metric is more practical in evaluating the synthetic images compared to other metrics like Inception score (IS) which fails to capture the semantic information hidden in the images.


<!-- ObjectPAthGAN Challenges -->
# Challenges
We faced few challeges in terms of model training due to the following reasons: 
1. OP-GAN model is big in terms of network size and thereby consuming huge amount of memory for weights. 
2. Train images are high resolution which further restricts the batch size to much lower value. (Usually GPU memory is 16GB)
3. Lesser batch size causes more time to model convergence. 
4. Discriminator may learn too fast/too slow and cause training instability

<!-- ObjectPAthGAN Improvements -->
# Improvements
1. We introduced Spectral normalization layer in discriminator to replace the 2D batch norm layer.
2. This constraints the [Lipschitz constant](https://arxiv.org/pdf/1802.05957.pdf) of the convolutional filters.
3. This helped to overcome the limitations of lower batch size and improved stability in discriminator.



<!-- ObjectPAthGAN High-level Archtecture -->
<br />
 <h3 align="center">High Level Architecture</h3>
    <img src="https://github.com/MasterProjectOPGAN/ObjectPathwayGAN/blob/main/Project-Screenshots/High-level%20Architecture.png" alt="homepage" width="950" height="500">

<br />
 <h3 align="center">Object Pathway</h3>
    <img src="https://github.com/MasterProjectOPGAN/ObjectPathwayGAN/blob/main/Project-Screenshots/Object%20Pathway.png" alt="Object Pathway" width="950" height="500">

<br />
 <h3 align="center">Global Pathway</h3>
    <img src="https://github.com/MasterProjectOPGAN/ObjectPathwayGAN/blob/main/Project-Screenshots/Global%20Pathway.png" alt="Global Pathway" width="950" height="500">
 
<!-- SOA -->
# Evaluation Metric Sematic Object Accuracy (SOA)
<br />
 <h3 align="center">SOA YOLO</h3>
    <img src="https://github.com/MasterProjectOPGAN/ObjectPathwayGAN/blob/main/Project-Screenshots/SOA-YOLO.png" alt="SOA yolo" width="950" height="500">

<br />
 <h3 align="center">SOA model comparison</h3>
    <img src="https://github.com/MasterProjectOPGAN/ObjectPathwayGAN/blob/main/Project-Screenshots/SOA%20-%20Model%20Comparison.png" alt="SOA model comparison" width="950" height="500">

<br />
 <h3 align="center">Comparison of SOA for OPGAN and Spectral OPGAN</h3>
    <img src="https://github.com/MasterProjectOPGAN/ObjectPathwayGAN/blob/main/Project-Screenshots/SOA%20Comparison.png" alt="Comparison of SOA for OPGAN and Spectral OPGAN" width="950" height="500">

<!-- MLFlow -->
# MLOps using MLFlow
<br />
 <h3 align="center">Generator Loss tracked using MLFlow</h3>
    <img src="https://github.com/MasterProjectOPGAN/ObjectPathwayGAN/blob/main/Project-Screenshots/MLops%20in%20MLFlow.png" alt="Generator Loss tracked using MLFlow" width="950" height="500">


<!-- Demo -->
# Demo 
<br />
 <h3 align="center">Demo showing generation of images at respective checkpoints</h3>
    <img src="https://github.com/MasterProjectOPGAN/ObjectPathwayGAN/blob/main/Project-Screenshots/WebApplication-screenshot.png" alt="Generator Loss tracked using MLFlow" width="950" height="500">




## To Use OP-GAN 
#### Dependencies
- python 3.8.5
- pytorch 1.7.1

Go to ``OP-GAN``.
Please add the project folder to PYTHONPATH and install the required dependencies:

```
conda env create -f environment.yml
```

#### Data
- MS-COCO:
    - [download](https://www2.informatik.uni-hamburg.de/wtm/software/semantic-object-accuracy/data.tar.gz) our preprocessed data (bounding boxes, bounding box labels, preprocessed captions), save to `data/` and extract
        - the preprocessed captions are obtained from and are the same as in the [AttnGAN implementation](https://github.com/taoxugit/AttnGAN)
        - the generateod bounding boxes for evaluating at test time were generated with code from the [Obj-GAN](https://github.com/jamesli1618/Obj-GAN)
    - obtain the train and validation images from the 2014 split [here](http://cocodataset.org/#download), extract and save them in `data/train/` and `data/test/`
    - download the pre-trained DAMSM for COCO model from [here](https://github.com/taoxugit/AttnGAN), put it into `models/` and extract

#### Training
- to start training run `sh train.sh gpu-ids` where you choose which gpus to train on
    - e.g. `sh train.sh 0,1,2,3`
- training parameters can be adapted via `code/cfg/dataset_train.yml`, if you train on more/fewer GPUs or have more VRAM adjust the batch sizes as needed
- make sure the DATA_DIR in the respective `code/cfg/cfg_file_train.yml` points to the correct path
- results are stored in `output/`

#### Evaluating
- update the eval cfg file in `code/cfg/dataset_eval.yml` and adapt the path of `NET_G` to point to the model you want to use (default path is to the pretrained model linked below)
- run `sh sample.sh gpu-ids` to generate images using the specified model
    - e.g. `sh sample.sh 0`

#### Pretrained Models
- OP-GAN: [download](https://www2.informatik.uni-hamburg.de/wtm/software/semantic-object-accuracy/op-gan.pth) and save to `models`


## Acknowledgement
- [OPGAN](https://arxiv.org/pdf/1910.13321v2.pdf) T. Hinz, S. Heinrich and S. Wermter, "Semantic Object Accuracy for Generative Text-to-Image Synthesis," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2020.3021209.
- Code and preprocessed metadata for the experiments on MS-COCO are adapted from [AttnGAN](https://github.com/taoxugit/AttnGAN) and [AttnGAN+OP](https://github.com/tohinz/multiple-objects-gan).
- Code to generate bounding boxes for evaluation at test time is from the [Obj-GAN](https://github.com/jamesli1618/Obj-GAN) implementation.
- Code for using YOLOv3 is adapted from [here](https://pjreddie.com/darknet/), [here](https://github.com/eriklindernoren/PyTorch-YOLOv3), and [here](https://github.com/ayooshkathuria/pytorch-yolo-v3).
