# krakencoder_paper
Analysis and figure code for Krakencoder publication:

Keith W. Jamison, Zijin Gu, Qinxin Wang, Mert R. Sabuncu, Amy Kuceyeski, "Release the Krakencoder: A unified brain connectome translation and fusion tool". bioRxiv [doi:10.1101/2024.04.12.589274](https://www.biorxiv.org/content/10.1101/2024.04.12.589274) 

Main Krakencoder tool here: [github.com/kjamison/krakencoder](https://github.com/kjamison/krakencoder)

# Code organization
### Jupyter notebooks
* [`demographic_prediction.ipynb`](demographic_prediction.ipynb): Jupyter notebook to run prediction models and prediction figures
* [`family_similarity_comparison.ipynb`](family_similarity_comparison.ipynb): Jupyter notebook to generate family group violin figures
### User-facing scripts
* [`run_connectome_performance_comparison.py`](run_connectome_performance_comparison.py): Script to run graph metrics on observed and predicted connectomes, edgewise accuracy, compare different prediction types, etc.
### Internal scripts
* [`predict.py`](predict.py): Functions for fitting and evaluating demographic predictions
* [`family_group_stats.py`](family_group_stats.py): Functions for computing and visualizing family group separability
* [`graph_measures.py`](graph_measures.py): Functions for computing graph metrics on observed and predicted connectomes
* [`data.py`](data.py): Functions for loading and manipulating Krakencoder data, raw data, demographic data, etc
* [`plotting.py`](plotting.py): Functions for plotting demographic prediction bar plots
* [`utils.py`](utils.py): Miscellaneous functions

# Requirements
* krakencoder: [github.com/kjamison/krakencoder](https://github.com/kjamison/krakencoder)
* python >= 3.8
* pytorch >= 1.10
* numpy >= 1.21.2
* scipy >= 1.7.2
* scikit_learn >= 0.23.2
* statsmodels >= 0.14
* bctpy >= 0.6.0 
* pandas >= 1.5
* matplotlib, seaborn, tqdm, ipython (ipywidgets, ipykernel)
* *See [`requirements_exact.txt`](requirements_exact.txt)*
<br><br>
* Uses Brain Connectivity Toolbox for Python 0.6.0 (bctpy): [github.com/aestrivex/bctpy](https://github.com/aestrivex/bctpy)
    * Original Matlab version and additional documentation: [https://sites.google.com/site/bctnet/](https://sites.google.com/site/bctnet/)