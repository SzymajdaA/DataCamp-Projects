
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

#looking form missing values and data types
crops.isna().sum()
crops['crop'].unique()
crops.dtypes

#encoding the classes

le = LabelEncoder()
label = le.fit_transform(crops['crop'])
crops.drop("crop", axis=1, inplace=True)
crops["crop"] = label

#splitting the data

X = crops.iloc[:,0:4]
y = crops.iloc[:,4]
crops.dtypes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 42)

#fitting model for each feature

for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(max_iter=2000, multi_class='multinomial')
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    feature_performance = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-score for {feature}: {feature_performance}")
    
#creating heatmap

crops.corr()
sns.heatmap(crops.corr(), annot = True)

#choosing best features

final_features = ['N', 'K', 'ph']

#final performance of the model

X = crops[final_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(max_iter=2000, multi_class='multinomial')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
model_performance = f1_score(y_test, y_pred, average='weighted')
print(f"F1-score: {model_performance}")