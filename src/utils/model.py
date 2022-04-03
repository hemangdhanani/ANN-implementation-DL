import tensorflow as tf
import time
import os

def create_model(optimizer_val, loss_func, matrics_method, num_classes):
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(num_classes, activation="softmax", name="outputLayer")]
    model_clf = tf.keras.models.Sequential(LAYERS) 
    # loss_func = "sparse_categorical_crossentropy"
    # optimizer_val = "SGD"
    # matrics_method = ["accuracy"]

    model_clf.compile(optimizer=optimizer_val, loss=loss_func,metrics=matrics_method)   
    return model_clf # untrained model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model) 