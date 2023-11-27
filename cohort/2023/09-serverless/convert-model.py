import tensorflow as tf
from tensorflow import keras

def convert_model(model_name: string, tflite_model_name: string):
  # Load Model
  model = keras.models.load_model(model_name)

  # Convert model into tflite format
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # write tflite model into file
  with open(tflite_model_name, 'wb') as f_out:
    f_out.write(tflite_model)

if __name__ == "__main__":
  model_name = "clothing-model.h5"
  tflite_model_name == "clothing-model.tflite"
  convert_model(model_name, tflite_model_name)
