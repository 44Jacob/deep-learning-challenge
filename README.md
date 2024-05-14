# deep-learning-challenge
Alphabet Soup Charity Deep Learning Challenge
Overview

The Alphabet Soup Charity Deep Learning Challenge aims to create a binary classifier to predict whether applicants will be successful if funded by Alphabet Soup. Using a dataset containing information about past funding recipients, we preprocess the data, build and train a neural network model, and optimize it to achieve the best predictive accuracy possible.

Project Structure
deep-learning-challenge/
AlphabetSoupCharity.ipynb: Initial notebook for data preprocessing, model training, and evaluation.
AlphabetSoupCharity_Optimization.ipynb: Notebook for model optimization attempts.
AlphabetSoupCharity.h5: HDF5 file containing the trained model.
AlphabetSoupCharity_Optimization.h5: HDF5 file containing the optimized model.
charity_data.csv: Dataset provided for the challenge.
README.md: This readme file.
Requirements
To run the notebooks and reproduce the results, you need the following Python packages:

pandas
numpy
scikit-learn
tensorflow
keras

You can install these dependencies using pip:

bash
Copy code
pip install pandas numpy scikit-learn tensorflow keras
Data Preprocessing

Load and Inspect Data:

Load the charity_data.csv into a Pandas DataFrame.
Identify target variable (IS_SUCCESSFUL) and feature variables.
Drop non-beneficial columns: EIN and NAME.
Analyze Categorical Data:

Determine the number of unique values in each column.
Replace rare categorical values with "Other" based on a chosen cutoff point.

Encode Categorical Data:
Use pd.get_dummies() to convert categorical data into numerical data.

Split and Scale Data:
Split the preprocessed data into training and testing datasets.
Scale the features using StandardScaler.
Model Building, Training, and Evaluation

Define the Model:
Use TensorFlow and Keras to build a Sequential model.
Add input, hidden, and output layers with appropriate activation functions.

Compile and Train the Model:
Compile the model using the Adam optimizer and binary cross-entropy loss function.
Train the model for 100 epochs.

Evaluate the Model:
Evaluate the model using the test data to determine its loss and accuracy.
Save the trained model to AlphabetSoupCharity.h5.
Model Optimization

Optimize the Model:
Create a new notebook (AlphabetSoupCharity_Optimization.ipynb) for optimization.

Implement at least three optimization methods:
Adjust the number of neurons and layers.
Experiment with different activation functions.
Adjust the number of epochs.

Evaluate the Optimized Model:
Evaluate the optimized model and save it to AlphabetSoupCharity_Optimization.h5.

Results and Report

Data Preprocessing
Target Variable: IS_SUCCESSFUL
Feature Variables: All columns except EIN and NAME
Removed Variables: EIN, NAME
Model Training and Evaluation

Initial Model:
Layers: 1 hidden layer with 10 neurons.
Activation Functions: ReLU for the hidden layer, Sigmoid for the output layer.
Accuracy: 73.24%

Optimized Model:
Layers: 2 hidden layers with 80 and 30 neurons.
Activation Functions: ReLU for hidden layers, Sigmoid for output layer.
Accuracy: 78.82%

Summary
The deep learning model built for the Alphabet Soup Charity achieves a satisfactory accuracy in predicting successful funding applications. Further optimization and experimentation with different model architectures and hyperparameters could potentially improve the model's performance.

Recommendations
For further improvement, consider exploring other machine learning models such as Random Forest or Gradient Boosting, which might provide better predictive performance on this classification problem.
