# Sketch-to-face
This project leverages the power of deep Generative Adversarial Networks to convert a hand sketched face into a real human face

Paper link: https://arxiv.org/abs/2008.00951
Official Repo Link: https://github.com/eladrich/pixel2style2pixel

System requirements:
OS: Linux/Mac OS
Software requirements: python3.5+, OpenCV, scikit-learn, numpy
Ninja compiler needs to be installed
Hardware used for training: Google Colab with 15 GB GPU – Nvidia Tesla T4.

CelebAHQdataset: [Kaggle Link](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)

The important contribution of the team is the script to generate synthetic sketch images using python and OpenCV. 
The Code for reference is in scripts/pencil_sketch_create_dataset.py

To run training/testing on CelebAHQ dataset using Google Colab follow the steps mentioned in the below link:
https://colab.research.google.com/drive/1YYNC-yscl2AA6nNJg7k35Re5b51g4jni?usp=sharing 

Training command for SketchtoFace Encoder:
python scripts/train.py \
--dataset_type=celebs_sketch_to_face \
--exp_dir=/content/drive/MyDrive/EEE511/Nov14_CelebA_Retrain_270M \
--checkpoint_path=/content/drive/MyDrive/EEE511/Nov9_CelebA_Retrain_270M/checkpoints/best_model.pt \
--workers=4 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=4 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0 \
--w_norm_lambda=0.005 \
--label_nc=1 \
--input_nc=1 

Model: 

Results for the given input:
![Uploading Screenshot (29).png…]()
![Uploading Screenshot (30).png…]()






