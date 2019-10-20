# Fast-SRGAN
The goal of this repository is to enable real time super resolution for upsampling low resolution videos. Currently, the design follows the [SR-GAN](https://arxiv.org/pdf/1609.04802.pdf) architecture. But instead of residual blocks, inverted residual blocks are employed from the MobileNet for parameter efficiency and fast operation. This idea is somewhat inspired by [Real time image enhancement GANs](http://www.micc.unifi.it/seidenari/wp-content/papercite-data/pdf/caip_2019.pdf).

The results are obviously not as good as the SRGAN, since this is a "weaker" generator. But it is faster. Benchmarks coming soon. Any ideas on impoving it/pull requests are welcome!

# Code
Code is written to be clean and readable. And is written in the tensorflow 2.0 style. Functions are decorated with tf.function where ever necessary.

# Pretrained model
A pretrained generator model on the DIV2k dataset is provided in the 'models' directory. It uses 12 inverted residual blocks, with 24 filters in every layer of the generator. Upsampling is done via phase shifts AKA pixel shuffle. During training pixel shuffle upsampling gave checkerboard artifacts. Adding MSE as a loss reduced them. I tried ICNR initialization, but that didn't seem to help as the artifacts would appear near the end.

# Training curves because why not?
<p align="center">
  <img src="https://user-images.githubusercontent.com/4294680/67163297-8df2df80-f36d-11e9-9517-3822b4f4105c.png"> <img src="https://user-images.githubusercontent.com/4294680/67163308-a662fa00-f36d-11e9-8f17-28ec6bde4ab9.png">
  <img src="https://user-images.githubusercontent.com/4294680/67163317-ba0e6080-f36d-11e9-936b-3579f4bb5d45.png"> <img src="https://user-images.githubusercontent.com/4294680/67163321-cabed680-f36d-11e9-9d0f-bae077e99b20.png">
</p>

# Training Speed
On a GTX 1080 with a batch size of 14 and image size of 128, the model trains in 9.5 hours for 170,000 iterations. This is achieved mainly by the efficient tensorflow tf Data pipeline. It keeps the utilization at a constant 95%+.

# Samples
Following are some results from the provided trained model. Left shows the 4x downsampled image AFTER it has been upsampled 4x by bicubic interpolation. Middle is the output of the model. Right is the actual high res image.
<p align="center">
  256x256 to 1024x1024 upsampling:
  <img src="https://user-images.githubusercontent.com/4294680/67163689-4fabef00-f372-11e9-9a39-87552792cd70.png"> 
  128x128 to 512x512 upsampling:
  <img src="https://user-images.githubusercontent.com/4294680/67163721-b03b2c00-f372-11e9-84d9-9774f3c52657.png">
  64x64 to 256x256 upsampling:
  <img src="https://user-images.githubusercontent.com/4294680/67163743-de207080-f372-11e9-843f-87b9a6aba632.png">
  32x32 to 128x128 upsampling:
  <img src="https://user-images.githubusercontent.com/4294680/67163760-04461080-f373-11e9-902d-89dc3acb6e7b.png">
</p>

# Changing Input Size
The provided model was trained on 128x128 inputs, but to run it on inputs of arbitrary size, you'll have to change the input shape like so:
```python
from tensorflow import keras

# Load the model
model = keras.models.load_model('models/generator.h5')

# Define arbitrary spatial dims, and 3 channels.
inputs = keras.Input((None, None, 3))

# Trace out the graph using the input:
outputs = model(inputs)

# Override the model:
model = keras.model.Model(inputs, outputs)

# Now you are free to predict on images of any size.
```

# Contributing
If you have ideas on improving the model performance, adding metrics, or any other changes, please make a pull request or open an issue. I'd be happy to accept any contributions.

# TODOs
1. Add metrics.
2. Add runtime benchmarks.
3. Improve quality of the generator.
4. Investigate pre-activation content loss and WGANs.
5. Bug Fixes.
