# Data
The raw recorded XML and CSV files are located in the ./expe_data folder.

# Dependencies
To run this project, the easiest way is to create a python 3.7 virtual environment using conda (or anaconda navigator). The virtual environment must contain recent versions of matplotlib, numpy, pandas, scipy and seaborn. (Tested in January 2020).

Using conda from the command line:

```
conda create --name my_virtual_env python=3.7
conda activate my_virtual_env
conda install matplotlib numpy pandas scipy seaborn
```

A proper dependency management using pipenv is planned but is not available yet.

# Usage
To run the data processing and display, please run the experiment4params.py script inside your virtual environment. You must change the current directory to this directory.

```
cd PATH_TO_THIS_DIRECTORY
python3 ./experiment4params.py
```

To have access to more graphs, uncomment lines in the main script: experiment4params.py

Further details are given in other .py source files.
