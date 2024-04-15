import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data\creditcard.csv")
correlação = df.corr()
df = df.drop_duplicates()

X = df.drop('Class', axis=1)
y = df['Class']

selection = VarianceThreshold(threshold=0.400)
X_selection = selection.fit_transform(X)

X_selection = pd.DataFrame(X_selection, columns=X.iloc[:, 0:24].columns)

X_train, X_test, y_train, y_test = train_test_split(X_selection, y, test_size=0.25,  random_state=3)