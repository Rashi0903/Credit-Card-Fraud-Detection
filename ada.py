#Ada Boost Classifier
#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

#importing dataset
data = pd.read_csv(r"/content/drive/MyDrive/credit card/creditcard.csv")

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
Fractional_value = len(fraud)/len(valid)
print(Fractional_value)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

#using Scikit-learn to split the data into training and testing sets
from sklearn.model_selection import train_test_split

# Assuming 'Class' is the target variable and the rest are features
xData = data.drop(columns=['Class'])
yData = data['Class']

#Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size = 0.2, random_state = 42)

#Building the AdaBoost Classifier (ADABOOST)
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost model creation
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(xTrain, yTrain)
# Prediction
yPred = ada.predict(xTest)

#Evaluating the classifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix

n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()

print("The model used is AdaBoost classifier")

acc = accuracy_score(yTest, yPred)      #accuracy
print("The accuracy is {}".format(acc))

prec = precision_score(yTest, yPred)     #precision
print("The precision is {}".format(prec))

rec = recall_score(yTest, yPred)           #recall
print("The recall is {}".format(rec))

f1 = f1_score(yTest, yPred)                 #f1-score
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(yTest, yPred)       #MCC
print("The Matthews correlation coefficient is {}".format(MCC))

#confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
