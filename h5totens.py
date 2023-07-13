import keras
from keras.models import load_model
from keras_to_caffe import keras_to_caffe

# Load the Keras model from the .h5 file
model = load_model('age_model_checkpoint.h5')

# Convert the model to Caffe
keras_to_caffe(model, 'age_model_checkpoint.prototxt', 'age_model_checkpoint.caffemodel')
