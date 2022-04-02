import tensorflow as tf

def get_data(validation_datasize):
    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()

    X_cv, X_train = X_train[:validation_datasize]/255., X_train[validation_datasize:]/255.
    y_cv, y_train = y_train[:validation_datasize], y_train[validation_datasize:]

    X_test = X_test/255.

    return (X_train, y_train), (X_cv, y_cv), (X_test, y_test)
