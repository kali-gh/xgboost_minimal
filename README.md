### Background
This package implements a basic XGBoost estimation for the Mushroom dataset
https://archive.ics.uci.edu/dataset/73/mushroom

### Overview
- Downloads the data
- One hot encodes all the features
- Runs estimation with XGBoost
- Reports AUC score on validation and test set

### Results
- With very weak learners as parametrized we get an AUC of 0.96 in validation and test
- With slightly deeper learners (depth = 5) we achieve an AUC of 1. 
