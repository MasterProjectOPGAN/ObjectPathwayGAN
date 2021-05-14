#!/bin/bash
GPU=$1
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONUNBUFFERED=1
if [ -z "$GPU" ]
then
      echo "Starting training on CPU."
else
      echo "Starting training on GPU ${GPU}."
fi
python3 -u code/main.py --cfg code/cfg/cfg_file_train.yml --max_objects 3 --resume /home/paperspace/Project/ObjectPathwayGAN/semantic-object-accuracy-for-generative-text-to-image-synthesis-master/OP-GAN/output/coco_glu-gan2_2021_04_06_19_01_43_9609
echo "Done."
