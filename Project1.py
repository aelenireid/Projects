import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn as sk

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score

def data_processing(df):
    # Extract the target variable (train_y) and features (df_X)
    y = df["Step"]
    X = df.drop(columns=["Step"])

# Scaling the features
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_data = scaler.transform(X)
    scaled_data_df = pd.DataFrame(scaled_data, columns=X.columns)
    print("processed data")
    
    return scaled_data_df, y




#STEP 1: data import into Dataframe 
df = pd.read_csv("Project 1 Data.csv")

#STEP 2: displays data distrubution 
sns.countplot(df, x = "Step")
plt.show()


#STEP 3: merely looking at the correlation between points in the corr matrix of training data
corr_matrix = (df.drop(columns=["Step"])).corr()
sns.heatmap(np.abs(corr_matrix))

#STEP 4: 20, 80 divided into training and test data. #prepare for shuffle data.
# Scaling the features

X = df.drop(columns=["Step"])
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
train_X = pd.DataFrame(scaled_data, columns=X.columns)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
#splitting to 20% and 80% data
for train_index ,test_index in split.split(df,df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
    

# Extract the target variable (train_y) and features (df_X)
train_y = strat_train_set["Step"]
X = strat_train_set.drop(columns=["Step"])


# Extract the target variable (test_y) and features (df_X)
test_y = strat_test_set["Step"]
X = strat_test_set.drop(columns=["Step"])



#FOR GRID SEARCH PARAMETERS AND MODELS
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# # model 1 random forest
# model1 = RandomForestClassifier(random_state = 10)

# params1 = {
#     'n_estimators': [5,15,30],
#     'max_depth': [None,2,4,6],
#     'min_samples_split': [2, 6, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }

# grid_search = GridSearchCV(model1, params1, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
# grid_search.fit(train_X, train_y)
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
# best_model1 = grid_search.best_estimator_



# #model 2 Support vector machine
# model2 = LogisticRegression(random_state= 10)

# params2 = {
#     'C': [1,2,3,4,5],
#     'max_iter': [1000,2000,3000,4000],
# }

# print("\nrunning grid search for LogisticRegression Model")
# grid_search = GridSearchCV(model2, params2, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
# grid_search.fit(train_X, train_y)
# best_params2 = grid_search.best_params_
# print("Best Hyperparameters:", best_params2)
# best_model2 = grid_search.best_estimator_



#model 3 Support vector machine
model3 = SVC(random_state= 10)

params = {
    'C': [2,4,6,8,10],
    'kernel': ['linear','rbf','poly','sigmoid'],
    'gamma': ['scale','auto'],
}

print("\nrunning grid search for SVC Model")
grid_search = GridSearchCV(model3, params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_model3 = grid_search.best_estimator_


#for performance analysis
def getScores(true,pred):
    print("Precision score: ", precision_score(true, pred, average= 'micro'))
    print("Accuracy score: ", accuracy_score(true, pred))
    print("F1 score: ",f1_score(true, pred, average= 'micro'))
    
    return None


#STEP 5: performance

#model 1
best_model1.fit(train_X,train_y)
model1_pred = best_model1.predict(test_X)

print("\n~~scores for random forest model~~\n")
getScores(test_y,model1_pred)


#model 2
best_model2.fit(train_X,train_y)
model2_pred = best_model2.predict(test_X)

print("\n~~scores for SVC model~~\n")
getScores(test_y,model2_pred)

    

#model 3
best_model3.fit(train_X,train_y)
model3_pred = best_model3.predict(test_X)

print("\n~~scores for DTC model~~\n")
getScores(test_y,model3_pred)



