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
