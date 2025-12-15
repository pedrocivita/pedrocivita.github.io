# 1. Data

## Deadline and Submission

- **Deadline:** 05.sep (friday) until 23:59
- **Individual submission**
- **Delivery:** GitHub Pages link via insper.blackboard.com

## Activity: Data Preparation and Analysis for Neural Networks

This activity tests your ability to generate synthetic datasets, handle real-world data challenges, and prepare data for neural networks.

### Exercise 1 – Exploring Class Separability in 2D
1. Generate 400 samples divided equally into 4 classes using Gaussian distributions:
   - Class 0: mean = (0,0), std = 0.3
   - Class 1: mean = (1,1), std = 0.3
   - Class 2: mean = (0,1), std = 0.3
   - Class 3: mean = (1,0), std = 0.3
2. Plot the data with a different color for each class.
3. Analyze the scatter plot:
   - Describe distribution and overlap of classes.
   - Could a simple linear boundary separate all classes?
   - Sketch the decision boundaries a neural network might learn.

### Exercise 2 – Non-Linearity in Higher Dimensions
1. Generate two 5D classes (A and B) with 500 samples each using multivariate normal distributions:
   - Class A: mean = (1,1,1,1,1), covariance = identity
   - Class B: mean = (-1,-1,-1,-1,-1), covariance = identity
2. Reduce dimensionality to 2D using PCA and plot the projection coloring each class.
3. Discuss the relationship between classes and linear separability. Explain why non-linear models are required.

### Exercise 3 – Preparing Real-World Data for a Neural Network
1. Download the Kaggle **Spaceship Titanic** dataset.
2. Describe the dataset and identify numerical and categorical features. Investigate missing values.
3. Preprocess the data:
   - Handle missing values with a justified strategy.
   - One-hot encode categorical features.
   - Standardize or normalize numerical columns for `tanh` activation.
4. Visualize distributions before and after scaling.

## Evaluation Criteria

| Exercise | Points | Description |
|----------|--------|-------------|
| 1 | 3 | Data generated and visualized; analysis of separability and boundaries |
| 2 | 3 | Data generated, PCA applied, and analysis of non-linearity |
| 3 | 4 | Dataset described, preprocessing justified, and visualizations included |

