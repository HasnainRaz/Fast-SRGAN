# Fast-SRGAN
The goal of this repository is to enable real time super resolution for upsampling low resolution videos. Currently, the design follows the [SR-GAN](https://arxiv.org/pdf/1609.04802.pdf) architecture. For speed, the upsampling is done through pixel shuffle.

The training setup looks like the following diagram:

<p align="center">
  <img src="https://user-images.githubusercontent.com/4294680/67164120-22157480-f377-11e9-87c1-5b6acace0e47.png">
</p>

# Speed Benchmarks
The following runtimes/fps are obtained by averaging runtimes over 800 frames. Measured on MPS (MacBook M1 Pro GPU).

| Input Image Size  |      Output Size     | Time (s)  | FPS |
|   -------------   |:--------------------:|:---------:|:---:|
|     90x160        |    360x640 (360p)    |   0.01    | 82  |
|     180x320       |    720x1080 (720p)   |   0.04    | 27  |

We see it's possible to upsample to 720p at around 30fps.

# Requirements
This was tested on Python 3.10. To install the required packages, use the provided Pipfile:
```bash
pip install pipenv --upgrade
pipenv install --system --deploy
```

# Pre-trained Model
A pretrained generator model on the DIV2k dataset is provided in the 'models' directory. It uses 8 residual blocks, with 64 filters in every layer of the generator. 


To try out the provided pretrained model on your own images, run the following:

```bash
python inference.py --image_dir 'path/to/your/image/directory' --output_dir 'path/to/save/super/resolution/images'
```

# Training
To train, simply edit the config file in the folder `configs/config.yaml` with your settings, and then launch the training with:
```bash
python train.py
```

You can also change the config parameters from the command line. The following will run training with a `batch_size` of 32, a generator with 12 residual blocks, and a path to the image directory `/path/to/image/dataset`.
```
python train.py data.image_dir="/path/to/image/dataset" training.batch_size=32 generator.n_layers=12

```
This is powered by `hydra`, which means all the parameters in the config are editable via the CLI.

Model checkpoints and training summaries are saved in tensorboard. To monitor training progress, open up tensorboard by pointing it to the `outputs` directory that will be created when you start training.

# Samples
Following are some results from the provided trained model. Left shows the low res image, after 4x bicubic upsampling. Middle is the output of the model. Right is the actual high resolution image.

<p align="center">
  <b>The following shows images upsampled 4x by bicubic interpolation, the pretrained model from this repository and the original high resolution image as a comparison</b>
  <img src="https://github.com/HasnainRaz/Fast-SRGAN/assets/4294680/95b6f8e4-f6c0-403b-854e-78c5589fbec6g"> 
  <img src="https://github.com/HasnainRaz/Fast-SRGAN/assets/4294680/d57abe02-46d8-48ce-bd1f-3cd06a9a66087">
  <img src="https://github.com/HasnainRaz/Fast-SRGAN/assets/4294680/67472974-56a5-4505-abaa-5e1c86467da1">
  <img src="https://github.com/HasnainRaz/Fast-SRGAN/assets/4294680/0d16647e-75ea-4150-bba0-2ea70ba05ca0">
</p>

# Contributing
If you have ideas on improving model performance, adding metrics, or any other changes, please make a pull request or open an issue. I'd be happy to accept any contributions.

