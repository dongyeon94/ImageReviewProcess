import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.Validation import ImageValidation

class ImageSimilarity(ImageValidation):
    """"""
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu=use_gpu)


def main(image, plot):
    runner = ImageSimilarity()
    ImageValidation = runner._EdgeDetection(image, plot)

    if ImageValidation:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',required=True)
    parser.add_argument('--plot')
    args = parser.parse_args()
    # print(parser.parse_args())
    main(args.img, args.plot)
    

