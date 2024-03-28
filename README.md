# Generating 3D Microstructure Images for O2 Fuel Cell Electrode using GANs Enhanced with Minkowski Functionals

This repository contains code related to the our research article titled "Generating 3D Microstructure Images for O2 Fuel Cell Electrode using GANs Enhanced with Minkowski Functionals". The code provided here enables the generation and training of 3D microstructure images using Generative Adversarial Networks (GANs) enhanced with Minkowski Functionals.

## Description

### input_datasets_3D.py
The `input_datasets_3D.py` file located in the `preprocess` directory is responsible for generating training 3D microstructure images in HDF5 format from a 750x750x750 3D image of the microstructure. These training images are of size of 64x64x64.

### main_train.py
The `main_train.py` file in the train directory is used for training the generative model. This script contains the main training loop and is responsible for optimizing the GAN parameters to generate realistic 3D microstructure images.

### SOFC_generate_twophase.py
The `SOFC_generate_twophase.py` file in the `postprocess` directory allows the user to utilize the trained model to generate synthetic 3D microstructure images. This script produces 3D microstructure images representing the O2 fuel cell electrode.

## Usage


## Citation
Soon

For more details, please refer to the research article or contact the authors.
