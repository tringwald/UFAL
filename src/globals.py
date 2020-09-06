import os
import os.path as osp

# Paths and folders
TENSORBOARD_RUN_DIR = '/cvhci/temp/tringwald/ufal/runs'
GENERIC_TMP_DIR = '/cvhci/temp/tringwald/ufal/tmp'
os.makedirs(TENSORBOARD_RUN_DIR, exist_ok=True)
os.makedirs(GENERIC_TMP_DIR, exist_ok=True)

PROJECT_ROOT = osp.abspath(osp.join(__file__, '../..'))

# Dataset variables
VISDA17_ROOT = 'data/VisDA-2017'
OFFICE_HOME_ROOT = 'data/Office-Home'
OFFICE_CALTECH_ROOT = 'data/Office-Caltech'
