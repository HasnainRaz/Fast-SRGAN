from argparse import ArgumentParser
from tensorflow import keras
import os
import tensorflow as tf

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')
  
def change_model(model, new_input_shape, custom_objects=None):
    # replace input shape of first layer    
    model.layers[0]._batch_input_shape = new_input_shape

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json(),custom_objects=custom_objects)

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model

def main():
    args = parser.parse_args()

    # Get all image paths
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]

    # Change model input shape to accept all size inputs
    model = keras.models.load_model('models/generator.h5')
    model = change_model(model, new_input_shape=[None, None, None, 3])

    # Loop over all images
    for image_path in image_paths:
        
        # Read image
        low_res = tf.image.decode_image(tf.io.read_file(image_path), channels = 3)

        # Rescale to 0-1.
        low_res = low_res / 255

        # Get super resolution image
        sr = model.predict(tf.expand_dims(low_res, axis=0))[0]

        # Rescale values in range 0-255
        sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

        # Save the results:
        if image_path.endswith(".png"):
            tf.io.write_file(os.path.join(args.output_dir, os.path.basename(image_path)), tf.image.encode_png(sr))
        elif image_path.endswith(".jpeg"):
            tf.io.write_file(os.path.join(args.output_dir, os.path.basename(image_path)), tf.image.encode_jpeg(sr))


if __name__ == '__main__':
    main()
