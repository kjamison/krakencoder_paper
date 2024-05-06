"""
Miscellaneous utility functions for the scripts in this repo.
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

def krakendir(dirtype='data'):
    """
    Returns the hardcoded path to the krakencoder directory.
    """
    if dirtype=='data':
        dirlist=["/Users/kwj5/Research/krakencoder",
                "/midtier/cocolab/colossus/shared_data3/kwj2001/HCP_connae",
                "/home/kwj2001/colossus_shared3/HCP_connae", 
                "/midtier/sablab/scratch/kwj2001/krakencoder"]
    elif dirtype=='source':
        dirlist=["/Users/kwj5/Source/krakencoder",
                 "/home/kwj2001/krakencoder"]
    else:
        raise Exception("Unknown dirtype: %s" % (dirtype))
    
    for d in dirlist:
        if os.path.exists(d):
            return d
    raise Exception("krakencoder directory not found")

def add_krakencoder_package(quiet=False):
    """
    Adds the krakencoder package to the path, if it is not already there.
    First checks if it is installed as a package. If not, checks for krakencoder source directory.
    """
    try:
        __import__('krakencoder.utils')
    except ModuleNotFoundError:
        krakencoder_sourcedir=krakendir('source')
        sys.path.append(krakencoder_sourcedir)
        if not quiet:
            print("Added %s to path" % (krakencoder_sourcedir))

def add_bctpy_package(quiet=False):
    """
    Adds the bctpy package to the path, if it is not already there.
    First checks if it is installed as a package. If not, checks for krakencoder source directory.
    """
    try:
        __import__('bct')
    except ModuleNotFoundError:
        dirlist=["/Users/kwj5/Source/bctpy",
                 "/midtier/cocolab/scratch/kwj2001/bctpy"]

        bctpy_dir=None
        for d in dirlist:
            if os.path.exists(d):
                bctpy_dir=d
        if bctpy_dir is None:
            raise Exception("BCTpy directory not found")
        
        sys.path.append(bctpy_dir)
        if not quiet:
            print("Added %s to path" % (bctpy_dir))

def iterable(x):
    """
    Returns True if x is iterable, False otherwise.
    """
    try:
        iter(x)
        return True
    except:
        return False

def normfun(x):
    """
    Normalize np.array x to have unit norm along axis 1.
    """
    if x is None:
        return None
    return x/np.sqrt(np.sum(x**2,axis=1,keepdims=True))

def timestamp():
    """
    Return a timestamp string in the format YYYYMMDD-HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def is_numeric_field(series):
    """
    Returns True if the pandas series can be converted to numeric, False otherwise.
    """
    try:
        pd.to_numeric(series)
        return True
    except ValueError:
        return False
