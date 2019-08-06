import os
import time
import random
import argparse
import h5py
import json
import math
import copy

import torch
import torch.nn as nn

from dataset.base_dataset import BaseDataset
from dataset.collate import UserScatteredDataParallel, user_scattered_collate
from dataset.dataloader import DataLoaderIter, DataLoader
from utils import AverageMeter, selective_load_weights

from model.builder import ModelBuilder
from model.base_model import BaseLearningModule
from model.parallel.replicate import patch_replication_callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

