# Multinomial Naive Bayes

John Warila, warilaj - Tuning and Implementation
Joshua Barringer, barrinjo - Algorithm design and Implementation

## Running instructions

Make sure to use python3 to run.

Required arguments
python3 imp2.py \<data file\> \<labels file\> \<run_type\>

The data and labels file should be two separate csv files.

**Important note**

When running testing data make sure to provide the full 50k labels and data!
There will be index out of bounds if you do not. For training and validation,
have the full 50k data, and 40k labels.

### Options for run_type argument

- validate_default , this runs the default settings and outputs the validation and training data. Predictions are saved in a temp training_validation.csv file
- validate_best , this runs the best settings and outputs teh validation and training data. Predictions are saved in a temp training_validation.csv file
- test_default , this runs the default settings on the testing data. Predictions go to test-predictions1.csv
- test_alpha , this runs the best alpha setting with the rest on default (note the best alpha happens to be default). Predictions are saved in test-predictions2.csv
- test_best , this runs the best settings on the test data. Predictions go to test-predictions3.csv
- cust , this runs a custom run. The optional arguments will be used. If none are provided this is the same as running validate_default.

### Optional argumnents

- \-\-alpha , this sets the alpha for laplace smoothing. Should be an int
- \-\-max_features , this sets the maximum features for the CountVectorizer
- \-\-max_df , this sets the maximum document frequency for the CountVectorizer
- \-\-min_df , this sets the minimum document frequency for the CountVectorizer

### Output format
In the command line the program will output the run_type and the accuracy.

In the output csv file there will be a single column of predictions. 1.0 is positive, 0.0 is negative