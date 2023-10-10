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
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
#splitting to 20% and 80% data
for train_index ,test_index in split.split(df,df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
    

# Extract the target variable (train_y) and features (df_X)
train_y = strat_train_set["Step"]
X = strat_train_set.drop(columns=["Step"])

# Scaling the features
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
train_X = pd.DataFrame(scaled_data, columns=X.columns)


# Extract the target variable (test_y) and features (df_X)
test_y = strat_test_set["Step"]
X = strat_test_set.drop(columns=["Step"])

# Scaling the features
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
test_X = pd.DataFrame(scaled_data, columns=X.columns)

