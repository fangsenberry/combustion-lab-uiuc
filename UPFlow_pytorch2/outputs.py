import sys
import os

# Add the utils directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from utils.fv_util import FlowInitialization as Fizi
from utils.fv_util import FlowAnalysis as Flay
from utils.fv_util import FlowConfig
import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from statsmodels.graphics.tsaplots import plot_acf
from scipy.fftpack import fft, ifft, fftfreq
from scipy.optimize import curve_fit
import random
from scipy.spatial import distance
from collections import deque
import time
import breakup_util as bu

from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

#############################################################################################
def main():
    # Define your experiment parameters
    config = FlowConfig(
        trial_path=r"D:\Spray_conditions\A03_512-complex-09-08-24.pth_inference\UPF_A03_C_DP_30_trial_1",
        img_path=r"D:\Spray_conditions\A03_512-complex-09-08-24.pth_inference\denoised_images",
        dir_ext=r'flow_npy\result_',
        step=1,
        custom_range='end',
        array_save_path='flow_data.h5',
        file_format='hdf5',
        image_save_range= 35
    )
    
    # Initialize FlowAnalysis with the configuration
    Fizi_config = Fizi(config)
    # Fizi_config.run_save_case()
    Fizi_config.run_gradient_calculations()
    
    # try:
    #     Fizi_config.run_gradient_calculations()
        
    # except Exception as e:
    #     print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
