# MOF-O2N2: 

To run the script place the MOF dataset in the folder and run the `main.py` script. In all scripts the `if __name__ == "__main__"` boilerplate code was intentionally left out to allow the scripts to be run by a simple import.

The `featureIteratedALL5groups.py` script iterates through all fetures as described by the user by separating the dataset into 5 equally-sized test sets. If the number of entries in the dataset is not a multiple of five, there may be upto 4 entries taht overlap between the last, and second-to-last test sets. 

Following this, the `StandardDeviationScreening.py` should be run to remove the entries with errors greater than 2 standard deviations of the entire set. The resulting datasets are placed in the `./2STD/` directory.

The `FeatureIteratedALL.py` can then be run to test entries of the `2STD` datasets using random train/test splitting. 
