# Machine Learning for EDS Data Decomposition 
Machine Learning Course Project 2

## Overview
Energy Dispersive X-ray Spectroscopy (EDS, EDX or XEDS) is a qualitative and quantitative X-ray microanalytical technique that can provide information on the chemical composition of a sample. The energies of the Characteristic X-rays allow the elements making up the sample to be identified, while the intensities of the Characteristic X-ray peaks allow the concentrations of the elements to be quantified. Given the EDS Data of a sample, this project aims to obtain the spectrum of the pure elements present in the sample as well as their relative concentrations at each pixel.

## Libraries Required and installation
1. [Hyperspy](https://hyperspy.org/): HyperSpy is an open source Python library which provides tools to facilitate the interactive data analysis of multi-dimensional datasets that can be described as multi-dimensional arrays of a given signal.To install, run the following commands in the terminal:
``` 
conda install hyperspy -c conda-forge 
conda install hyperspy-gui-traitsui -c conda-forge
```
2. VCA: The vertex component analysis (VCA) is a method for unsupervised endmember extraction from hyperspectral data assuming that the data is a linear mixture of pure components (spectra). To run VCA, follow [this](https://github.com/Laadr/VCA) link and download the ```VCA.py``` file to include in the same folder.
3. [Fuzzy-c-means](https://pypi.org/project/fuzzy-c-means/): Fuzzy-c-means is a Python module implementing the Fuzzy C-means clustering algorithm. To install, run the following:
```
pip install fuzzy-c-means
```
Apart from the above, a working version of the latest [scikit-learn](https://scikit-learn.org/stable/) for Python is required.

## Running the code.
1. After installing the above packages, the python notebook run.py contains the blocks of codes to run the relevant experiments.

