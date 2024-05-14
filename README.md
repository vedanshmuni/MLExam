# MLExam

EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
import geopandas as gpd
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
dataset = pd.read_csv('data.csv')

# Understanding the data
print(dataset.head())
print(dataset.info())
print(dataset.describe())

# Handling missing values
print(dataset.isnull().sum())
dataset.fillna(dataset.mean(), inplace=True)

# Exploratory Data Analysis
numerical_features = dataset.select_dtypes(include=[float, int]).columns
categorical_features = dataset.select_dtypes(include=[object]).columns

# Univariate Analysis
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=dataset, x=feature, kde=True)

for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=dataset, x=feature)

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=dataset, y=feature)

# Bivariate Analysis
sns.pairplot(dataset[numerical_features], diag_kind='kde')

plt.figure(figsize=(12, 10))
sns.heatmap(dataset[numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Multivariate Analysis
sns.pairplot(dataset)
plt.figure(figsize=(12, 8))
parallel_coordinates(dataset, class_column='target')
plt.show()

# Distribution Analysis
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=dataset, x=feature)

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=dataset, x=feature, y='target', inner='quartile')

# Geospatial Analysis
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.plot(column='gdp_md_est', cmap='OrRd', legend=True)
plt.title('Global GDP')
plt.show()

# Time Series Analysis
time_series_data = pd.read_csv('time_series_data.csv')
plt.figure(figsize=(10, 6))
sns.lineplot(data=time_series_data, x='date', y='value')

result = seasonal_decompose(time_series_data['value'], model='multiplicative', period=12)
result.plot()
plt.show()

# Feature Engineering
dataset['new_feature'] = dataset['feature1'] + dataset['feature2']
scaler = StandardScaler()
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

# Handling Categorical Features
dataset = pd.get_dummies(dataset, columns=categorical_features)

# Dimensionality Reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(dataset[numerical_features])
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Outlier Detection
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=dataset[feature])
    plt.title("Boxplot of " + feature)
    plt.show()


CLEANING

import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import re
from nltk.tokenize import word_tokenize

# Load data
dataset = pd.read_csv('your_data.csv')

# 1. Handling Missing Values
missing_values = dataset.isnull().sum()
dataset.fillna(dataset.mean(), inplace=True)

# 2. Outlier Detection and Handling
z_scores = stats.zscore(dataset.select_dtypes(include=['float64', 'int64']))
outliers = (np.abs(z_scores) < 3).all(axis=1)
dataset = dataset[outliers]

# 3. Data Transformation
scaler = StandardScaler()
numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
dataset[numerical_cols] = scaler.fit_transform(dataset[numerical_cols])

# 4. Handling Duplicate Data
dataset.drop_duplicates(inplace=True)

# 5. Feature Engineering
dataset['total_income'] = dataset['income'] + dataset['bonus']

# 6. Handling Skewed Data
dataset['skewed_column'] = np.log(dataset['skewed_column'] + 1)  # Adding 1 to handle zeros

# 7. Handling Data Types
dataset['date_column'] = pd.to_datetime(dataset['date_column'])

# 8. Handling Text Data
dataset['text_column'] = dataset['text_column'].apply(lambda x: re.sub(r'\W', ' ', x.lower()))
dataset['text_tokens'] = dataset['text_column'].apply(word_tokenize)

# 9. Handling Time Series Data
dataset['day_of_week'] = dataset['date_column'].dt.dayofweek

# Now, your cleaned data is in the DataFrame 'dataset'


MODELING

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('your_dataset.csv')

# Split dataset into features and target variable
X = dataset.drop('target_column', axis=1)
y = dataset['target_column']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression, Lasso, Ridge
linear_reg = LinearRegression()
lasso = Lasso()
ridge = Ridge()

# Fit models
linear_reg.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)

# Predictions
linear_reg_preds = linear_reg.predict(X_test_scaled)
lasso_preds = lasso.predict(X_test_scaled)
ridge_preds = ridge.predict(X_test_scaled)

# Evaluation
print("Linear Regression:")
print("R^2 Score:", r2_score(y_test, linear_reg_preds))
print("Mean Absolute Error:", mean_absolute_error(y_test, linear_reg_preds))
print("Mean Squared Error:", mean_squared_error(y_test, linear_reg_preds))


print("Lasso:")
print("Mean Squared Error:", mean_squared_error(y_test, lasso_preds))
print("R^2 Score:", r2_score(y_test, lasso_preds))

print("Ridge:")
print("Mean Squared Error:", mean_squared_error(y_test, ridge_preds))
print("R^2 Score:", r2_score(y_test, ridge_preds))

# 2. Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
decision_tree_preds = decision_tree.predict(X_test)

# Overfitting graph
train_errors = []
test_errors = []
for depth in range(1, 20):
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    train_errors.append(1 - dt.score(X_train, y_train))
    test_errors.append(1 - dt.score(X_test, y_test))
    
plt.plot(range(1, 20), train_errors, label='Train')
plt.plot(range(1, 20), test_errors, label='Test')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Error')
plt.legend()
plt.show()

# Evaluation
print("Decision Tree:")
print("Accuracy Score:", accuracy_score(y_test, decision_tree_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, decision_tree_preds))
print("Classification Report:\n", classification_report(y_test, decision_tree_preds))

# 3. Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
naive_bayes_preds = naive_bayes.predict(X_test)

# Evaluation
print("Naive Bayes:")
print("Accuracy Score:", accuracy_score(y_test, naive_bayes_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, naive_bayes_preds))
print("Classification Report:\n", classification_report(y_test, naive_bayes_preds))

# 4. Logistic Regression with Hyperparameter Tuning
logistic_reg = LogisticRegression()
params = {'C': [0.1, 1, 10, 100]}
logistic_reg_grid = GridSearchCV(logistic_reg, params, cv=5)
logistic_reg_grid.fit(X_train_scaled, y_train)
logistic_reg_best = logistic_reg_grid.best_estimator_
logistic_reg_best.fit(X_train_scaled, y_train)
logistic_reg_preds = logistic_reg_best.predict(X_test_scaled)

# Evaluation
print("Logistic Regression:")
print("Accuracy Score:", accuracy_score(y_test, logistic_reg_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_reg_preds))
print("Classification Report:\n", classification_report(y_test, logistic_reg_preds))

# 5. SVM Linear and Kernel with Hyperparameter Tuning
svm_linear = SVC(kernel='linear')
svm_kernel = SVC(kernel='rbf')
params_svm = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}
svm_grid = GridSearchCV(svm_kernel, params_svm, cv=5)
svm_grid.fit(X_train_scaled, y_train)
svm_best = svm_grid.best_estimator_
svm_best.fit(X_train_scaled, y_train)
svm_preds = svm_best.predict(X_test_scaled)

# Evaluation
print("SVM:")
print("Accuracy Score:", accuracy_score(y_test, svm_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_preds))
print("Classification Report:\n", classification_report(y_test, svm_preds))

# 6. Random Forest with Feature Importance
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
feature_importances = random_forest.feature_importances_

# Evaluation
print("Random Forest:")
print("Accuracy Score:", accuracy_score(y_test, random_forest.predict(X_test)))
print("Feature Importances:", feature_importances)

# 7. AdaBoost with Hyperparameter Tuning
adaboost = AdaBoostClassifier()
params_adaboost = {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.1, 1]}
adaboost_grid = GridSearchCV(adaboost, params_adaboost, cv=5)
adaboost_grid.fit(X_train, y_train)
adaboost_best = adaboost_grid.best_estimator_
adaboost_best.fit(X_train, y_train)
adaboost_preds = adaboost_best.predict(X_test)

# Evaluation
print("AdaBoost:")
print("Accuracy Score:", accuracy_score(y_test, adaboost_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, adaboost_preds))
print("Classification Report:\n", classification_report(y_test, adaboost_preds))

# 8. XGBoost with Hyperparameter Tuning
xgb = XGBClassifier()
params_xgb = {'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7, 9]}
xgb_grid = GridSearchCV(xgb, params_xgb, cv=5)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_best.fit(X_train, y_train)
xgb_preds = xgb_best.predict(X_test)

# Evaluation
print("XGBoost:")
print("Accuracy Score:", accuracy_score(y_test, xgb_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))
print("Classification Report:\n", classification_report(y_test, xgb_preds))

from sklearn.metrics import silhouette_score, adjusted_rand_score

# 9. KMeans Clustering with Pipeline
from sklearn.pipeline import make_pipeline
kmeans = KMeans(n_clusters=2)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(X_train)
cluster_preds = pipeline.predict(X_test)

# Evaluation
silhouette = silhouette_score(X_test_scaled, cluster_preds)
adjusted_rand = adjusted_rand_score(y_test, cluster_preds)
print("KMeans Clustering:")
print("Silhouette Score:", silhouette)
print("Adjusted Rand Score:", adjusted_rand)

# 10. Comparison of any two algorithms on the same dataset
# Let's compare Logistic Regression and Random Forest
logistic_reg_score = accuracy_score(y_test, logistic_reg_preds)
random_forest_score = accuracy_score(y_test, random_forest.predict(X_test))
print("Logistic Regression Accuracy:", logistic_reg_score)
print("Random Forest Accuracy:", random_forest_score)

# 11. Model Building with Cross Validation
# Using cross_val_score
scores = cross_val_score(logistic_reg, X, y, cv=5)
print("Cross-validated Logistic Regression scores:", scores)

# 12. Linear Regression from Scratch
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.train_errors = []
        self.test_errors = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate training error
            train_error = np.mean((y_predicted - y) ** 2)
            self.train_errors.append(train_error)
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Usage:
linear_reg_scratch = LinearRegressionScratch()
linear_reg_scratch.fit(X_train_scaled, y_train)
linear_reg_scratch_preds = linear_reg_scratch.predict(X_test_scaled)

# Evaluation
print("Linear Regression from Scratch:")
print("Mean Squared Error:", mean_squared_error(y_test, linear_reg_scratch_preds))
print("R^2 Score:", r2_score(y_test, linear_reg_scratch_preds))




gradient



def cost_func(X, y, theta):
  m = len(y)
  y_pred = X.dot(theta)
  error = (y_pred - y) ** 2

  return 1/(2*m) * np.sum(error)
  
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        y_pred = X.dot(theta)
        error = np.dot(X.transpose(), (y_pred - y))
        theta -= alpha * 1 / m * error
        costs.append(cost_func(X, y, theta))
    return theta, costs

def predict(X, theta):
	y_pred = np.dot(theta.transpose(), X)
	return y_pred
