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



#for performance analysis
def getScores(true,pred):
    print("Precision: ", precision_score(true, pred, average= 'micro'))
    print("Accuracy: ", accuracy_score(true, pred))
    print("F1 record: ",f1_score(true, pred, average= 'micro'))
    
    
    return None


#STEP 1: data import into Dataframe 
df = pd.read_csv("Project 1 Data.csv")

#STEP 2: data distrubution is shown
sns.countplot(df, x = "Step")
plt.show()

#STEP 3: merely looking at the correlation between points in the corr matrix of training data
corr_matrix = (df.drop(columns=["Step"])).corr()
sns.heatmap(np.abs(corr_matrix))