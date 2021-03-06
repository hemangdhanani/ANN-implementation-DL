# from src.utils.common import read_config
import os
from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model
from utils.model import save_model

import argparse

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_cv, y_cv), (X_test, y_test) = get_data(validation_datasize)

    optimizer_val = config["params"]["optimizer"]
    loss_func = config["params"]["loss_function"]
    matrics_method = config["params"]["metrics"]
    num_classes = config["params"]["num_classes"]    

    model = create_model(optimizer_val, loss_func, matrics_method, num_classes)

    EPOCHS = config["params"]["epochs"]
    Validation = (X_cv, y_cv)
    history = model.fit(X_train, y_train,epochs=EPOCHS, validation_data=Validation)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    
    model_name = config["artifacts"]["model_name"]

    save_model(model, model_name, model_dir_path)

    # print(config)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)