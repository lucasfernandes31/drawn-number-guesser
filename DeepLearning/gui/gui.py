import os
from tkinter import Tk, Label, Button

import pygame
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

from DeepLearning.nn import Net, transform

WHITE = (255,255,255)
BLACK = (0,0,0)

class GUI:
    def __init__(self, model_path, image_size=(96,96)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net(10.to(self.device))