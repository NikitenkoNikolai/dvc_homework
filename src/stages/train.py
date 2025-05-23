import sys
import os
from pandas._config import config
sys.path.append(os.getcwd())
from src.loggers import get_logger
from prepare_dataset import load_config
from src.model_scripts.train import train
# from src.model_scripts.plot_model import vis_weigths

if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    train(config)
    