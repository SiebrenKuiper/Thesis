import numpy as np

BASE_IMG_DIR = r"D:/Research/MS_Thesis/Study2/Output/Image"
SKIP_LIST = [1, 22, 26, 28, 39, 41, 64, 67, 56]

HRF_MODEL = "glover" 
DRIFT_MODEL = "cosine" 
HIGH_PASS =   0.01
NOISE_MODEL = "ar1"
SMOOTHING_FWHM = 6
OVERSAMPLING = 100
N_JOBS = 1
MINIMIZE_MEMORY = True
VERBOSE = 2 

CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
    "rot_x", "rot_y", "rot_z",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
    
    "csf", "white_matter", "global_signal", 
    "framewise_displacement",
    "t_comp_cor_00", "t_comp_cor_01", "t_comp_cor_02", "t_comp_cor_03",
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03",
]


CONFOUND_MOTION = [
    "trans_x", "trans_y", "trans_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
    "rot_x", "rot_y", "rot_z",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2"
]
