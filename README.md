Classification on Imbalanced Insurance Claims Dataset

Project Purpose:-
The goal of this project is to build a machine learning model that can classify insurance claims as approved (1) or not approved (0).
The dataset is highly imbalanced, meaning one class (claims not approved) dominates the other. This imbalance can bias models toward predicting the majority class, so special handling is required.

Steps Taken :-
1. Data Exploration
Loaded the dataset and inspected its structure (head(), info(), describe())
Identified categorical and numerical columns.
Visualized distributions of numerical features and the target variable (claim_status).
Found that the target variable was imbalanced (majority class far larger than minority).

2. Handling Class Imbalance
Separated majority and minority classes.
Applied oversampling using sklearn.utils.resample to balance the dataset.
After resampling, both classes had equal representation, ensuring fairer training.

3. Feature Engineering
Separated features (X) and target (y).
Applied OneHotEncoding (OHE) to categorical features so they could be used 	in the model.
Scaling was skipped because Random Forest is tree-based and does not 	require scaled inputs.

4. Feature Importance
Trained a preliminary RandomForestClassifier on the encoded data.
Extracted feature importances to identify which features contributed most 	to predictions.
Selected the top 20 features to reduce dimensionality and speed up 	training.

5. Model Training
Split the data into training and test sets (train_test_split) with   	stratification to preserve class balance.
Trained a RandomForestClassifier on the selected features.
Evaluated the model using:
  Accuracy Score
  Classification Report (precision, recall, F1-score)
  Confusion Matrix

6. Visualization
Plotted a pie chart showing the proportion of correct vs incorrect predictions.
This provided a clear visual representation of model accuracy.

Results:
Accuracy Score: ~97%
Classification Report: Showed strong precision and recall for both classes, confirming balanced performance.
Confusion Matrix: Demonstrated that the model correctly classified most claims.

Pie Chart: Illustrated that the vast majority of predictions were correct.
