# Credit-Card-Fraud-Detection


This project demonstrates the application of machine learning techniques for detecting fraudulent credit card transactions. The dataset used is the popular credit card fraud dataset, which contains transactions made by credit cards in September 2013 by European cardholders.

## Dataset

The dataset used in this project is given . It contains 284,807 transactions, of which 492 are fraudulent. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

## Project Structure

- `Random_Forest_Classifier.ipynb`: Colab  NoteBook implementing the Random Forest classifier.
- `AdaBoost_Classifier.ipynb`: Colab NoteBook implementing the AdaBoost classifier.
- `data/creditcard.csv`: The dataset file (should be downloaded separately).

## Libraries Used

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Steps Followed

### 1. Data Preprocessing

- Load the dataset using pandas.
- Separate the dataset into fraud and valid transactions.
- Calculate the fraction of fraud cases.

### 2. Exploratory Data Analysis (EDA)

- Visualize the correlation matrix to understand relationships between features (This step is mentioned in the code but not implemented).

### 3. Data Splitting

- Split the data into training and testing sets using `train_test_split` from scikit-learn.

### 4. Model Building

Two models were built and evaluated:

#### 4.1 Random Forest Classifier

- Import the `RandomForestClassifier` from scikit-learn.
- Create and fit the Random Forest model.
- Predict the results on the test set.
- Evaluate the model using accuracy, precision, recall, F1-score, Matthews correlation coefficient, and confusion matrix.

#### 4.2 AdaBoost Classifier

- Import the `AdaBoostClassifier` from scikit-learn.
- Create and fit the AdaBoost model.
- Predict the results on the test set.
- Evaluate the model using accuracy, precision, recall, F1-score, Matthews correlation coefficient, and confusion matrix.

### 5. Evaluation Metrics

The following metrics were used to evaluate the models:

- **Accuracy**: The proportion of true results among the total number of cases examined.
- **Precision**: The proportion of true positive results in the predicted positive results.
- **Recall**: The proportion of true positive results in the actual positive results.
- **F1-Score**: The harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC)**: A measure of the quality of binary classifications.
- **Confusion Matrix**: A table used to describe the performance of a classification model.

## Results

### Random Forest Classifier

--The accuracy is 0.9995786664794073

--The precision is 0.9868421052631579

--The recall is 0.7653061224489796

--The F1-Score is 0.8620689655172413

--The Matthews correlation coefficient is0.8688552993136148

![download](https://github.com/user-attachments/assets/c3a15e79-7680-4695-b308-b1d65f79edb5)



### AdaBoost Classifier


--The accuracy is 0.9993153330290369

--The precision is 0.8554216867469879

--The recall is 0.7244897959183674

--The F1-Score is 0.7845303867403315

--The Matthews correlation coefficient is 0.7869053021234045

![download](https://github.com/user-attachments/assets/605c6924-138b-4bc4-8af9-8fb10151d798)


### Confusion Matrices

Confusion matrices for both classifiers were visualized using seaborn heatmaps.

## Conclusion

Both the Random Forest and AdaBoost classifiers performed well on the highly imbalanced dataset. However, the exact performance metrics and which model performed better can be determined by comparing the evaluation metrics.
