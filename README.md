import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
class LinearRegressionCustom:
    def __init__(self):
        self.model = LinearRegression()
    def fit(self, x, y):
        self.model.fit(x, y)
    def predict(self, x):
        return self.model.predict(x)
# Read the data
df = pd.read_csv('/content/credit_data.csv')
# Display basic information about the dataframe
df.head()
df.isnull().sum()
df.shape
# Analyze the distribution of transaction types using a pie chart
type_counts = df['type'].value_counts()
transactions = type_counts.index
quantity = type_counts.values
px.pie(df, values=quantity, names=transactions, hole=0.4, title="Distribution of Transaction Type")
# Drop rows with missing values
df = df.dropna()
# Map 'isFraud' column to categorical values
df['isFraud'] = df['isFraud'].map({0: 'No Fraud', 1: 'Fraud'})
# Map 'type' column to numerical values
df['type'] = df['type'].map({'PAYMENT': 1, 'TRANSFER': 4, 'CASH_OUT': 2, 'DEBIT': 5, 'CASH_IN': 3})
# Display unique values and value counts for the 'type' column
df['type'].unique()
df['type'].value_counts()
# Prepare features (x) and target variable (y)
x = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = df['isFraud']
# Encode categorical labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y_encoded, test_size=0.20, random_state=42)
# Train a Decision Tree model
model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(xtrain, ytrain)
accuracy_decision_tree = model_decision_tree.score(xtest, ytest)
print("Decision Tree Model Accuracy:", accuracy_decision_tree)
# Use the trained Decision Tree model to make predictions
prediction_decision_tree = model_decision_tree.predict([[4, 180, 181, 10]])
print("Decision Tree Prediction:", prediction_decision_tree)
# Train a custom Linear Regression model
model_linear_regression_custom = LinearRegressionCustom()
model_linear_regression_custom.fit(xtrain, ytrain)
# Use the custom Linear Regression model to make predictions
prediction_linear_regression_custom = model_linear_regression_custom.predict(xtest)
print("Linear Regression Custom Predictions:", prediction_linear_regression_custom)
# Evaluate various models using cross-validation
models = [
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC()),
]
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = model_selection.cross_val_score(model, x, y_encoded, cv=kfold, scoring='accuracy')
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
