
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


# Models from Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree



"""### Import data"""

df_raw = pd.read_csv("Size_DB.csv")
print(df_raw)

print(df_raw.info())

"""### Exploratory data analysis"""

print(df_raw.describe())

# Number of occurences for each size (target variable)-console
print(df_raw["size"].value_counts())

# Number of occurences for each size (target variable)-chart description
sns.countplot(x=df_raw["size"])
plt.show()
"""Note: Size `M` is the most common"""

# Age distribution
sns.displot(df_raw["age"])
plt.show()

"""Large fraction of population seems to be around the ages of `25 to 35 years old`"""

# Weight distribution
sns.displot(df_raw["weight"])
plt.show()

# height distribution
sns.displot(df_raw["height"])
plt.show()

"""Population weight and height seem to show reasonable normal distributions

### Removing outliers (z-score)
"""

# Removing Outliers
dfs = []
sizes = []

for size_type in df_raw['size'].unique():
    sizes.append(size_type)
    ndf = df_raw[['age', 'height', 'weight']][df_raw['size'] == size_type]
    zscore = ((ndf - ndf.mean()) / ndf.std())
    dfs.append(zscore)

for i in range(len(dfs)):
    dfs[i]['age'] = dfs[i]['age'][(dfs[i]['age'] > -3) & (dfs[i]['age'] < 3)]
    dfs[i]['height'] = dfs[i]['height'][(dfs[i]['height'] > -3) & (dfs[i]['height'] < 3)]
    dfs[i]['weight'] = dfs[i]['weight'][(dfs[i]['weight'] > -3) & (dfs[i]['weight'] < 3)]

for i in range(len(sizes)):
    dfs[i]['size'] = sizes[i]
df_raw = pd.concat(dfs)
print(df_raw.head())

"""### Filling missing data"""

# Check for missing values
df_raw.isna().sum()

# Filling missing data
df_raw["age"] = df_raw["age"].fillna(df_raw['age'].median())
df_raw["height"] = df_raw["height"].fillna(df_raw['height'].median())
df_raw["weight"] = df_raw["weight"].fillna(df_raw['weight'].median())

# Mapping clothes size from strings to numeric
df_raw['size'] = df_raw['size'].map({"XXS": 1,
                                     "S": 2,
                                     "M": 3,
                                     "L": 4,
                                     "XL": 5,
                                     "XXL": 6,
                                     "XXXL": 7})

# Check for missing values
df_raw.isna().sum()
print(df_raw)

"""### 
We will create new feature to help model training effectiveness:
* `bmi` (body-mass index) - medically accepted measure of obesity
"""

df_raw["bmi"] = df_raw["height"] / df_raw["weight"]

print(df_raw)

"""### Correlation matrix"""

corr = sns.heatmap(df_raw.corr(), annot=True)
plt.show()

"""Clothing `size` seems much more highly dependent on `weight` than `age` or `height`, and seems to be have a strong inverse correlation with `bmi`

### Splitting data into training and validation datasets
The target variable is clothing `size`, and we will let the validation set be 30% of the total population.
"""

# Features
X = df_raw.drop("size", axis=1)

# Target
y = df_raw["size"]

X.head()

y.head()

# Splitting data into training set and validation set

X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.30)

len(X_train), len(X_test)

"""### Training Model
We will try:
* Logistic Regression
* Decision Tree Classifier
"""

# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
          "Decision Tree": DecisionTreeClassifier()}


# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
   Fits and evaluates given machine learning models.
   models: a dict of different Scikit_Learn machine learning models
   X_train: training data (no labels)
   X_test: testing data (no labels)
   y_train: training labels
   y_test: test labels
   """
    # Set random seed
    np.random.seed(18)
    # Make a dictionary to keep model scores
    model_scores = {}
    model_coplexity = {}
    # Loop through models
    for name, model in models.items():
        # Fit model to data
        start = time.time()
        model.fit(X_train, y_train)
        # Evaluate model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
        end = time.time()
        model_coplexity[name] = end - start
    print('model_coplexity runtime:', end=' ')
    print(model_coplexity)
    return model_scores


model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)

print(model_scores)

model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar()
print(model_compare)


"""### Model evaluation
The DecisionTreeClassifier model scored highest in initial tests with `99.9749%` accuracy.
"""
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
start = time.time()
y_pred = model.predict(X_test)
end = time.time()
print(y_pred)
print('Decision tree predict runtime:', end=' ')
print(end-start)

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Classification report
print(classification_report(y_test, y_pred))

model1 = LogisticRegression()
model1.fit(X_train, y_train)
start = time.time()
y_pred1 = model.predict(X_test)
end = time.time()
print('Logistic regression predict runtime:', end=' ')
print(end-start)


# Build Decision Tree
# Limit of poor quality of graph representation due to present to depth 1
clf = tree.DecisionTreeClassifier(max_depth=1)
clf = clf.fit(X_test, y_pred)
tree.plot_tree(clf)
plt.show()

# Order of the features
print(df_raw.columns)
# ['age', 'height', 'weight', 'size', 'bmi']
"""### x[2]=weight
the feature in the root is weight
"""

"""### Conclusion

The trained model shows a weighted average accuracy of `99.9%`, so the evaluation metric of >95% has been met.

### Feature Importance
"""

# Find feature importance of ideal model
len(model.feature_importances_)

model.feature_importances_


# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))
    # Plot dataframe
    fix, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature Importance")
    ax.invert_yaxis()


plot_features(X_train.columns, model.feature_importances_)
plt.show()
"""`weight` seems to be an extremely significant determinant for the model relative to the other features"""
