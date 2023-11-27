import tensorflow.lite as tflite
# import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
import numpy as np
from PIL import Image

MODEL_NAME = os.getenv('MODEL_NAME', 'bees-wasps-v2.tflite')

def download_image(url):
  with request.urlopen(url) as resp:
    buffer = resp.read()
  stream = BytesIO(buffer)
  img = Image.open(stream)
  return imgs


def prepare_image(img, target_size):
  if img.mode != 'RGB':
      img = img.convert('RGB')
  img = img.resize(target_size, Image.NEAREST)
  return img

def prepare_input(x):
  return x / 255.0


def predict(url):
  image = download_image(url)
  target_size = (150, 150)
  prepared_image = prepare_image(image, target_size)

  x = np.array(prepared_image, dtype='float32')
  X = np.array([x])
  X = prepare_input(X)
  print("X --- ", X)

  interpreter = tflite.Interpreter(model_path=MODEL_NAME)
  interpreter.allocate_tensors()

  input_index = interpreter.get_input_details()[0]['index']
  output_index = interpreter.get_output_details()[0]['index']

  interpreter.set_tensor(input_index, X)
  interpreter.invoke()

  preds = interpreter.get_tensor(output_index)
  print("preds --- ", preds)

  return float(preds[0, 0])
  # return {"msg": "Hello world"}


def lambda_handler(event, context):
  url = event['url']
  print("url ---- ", url)
  pred = predict(url)
  result = {
      'prediction': pred
  }

  return result

