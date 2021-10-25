# This is the code for creating an sklearn model
# The model will be saved into an h5 file.
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=1e5)
df = pd.read_csv("New_Drowsy.csv")
X = []
y = []
l = df.loc[:,df.columns!='Status']
# print(l)
# print(df['Status'])
y= df['Status'].tolist()
X_train,X_test,Y_train,Y_test = train_test_split(l,y,test_size = 0.2, shuffle = True, random_state=2)
print(X_test,X_test.shape)
logreg.fit(l, y)
y_pred = logreg.predict(X_test)
score = accuracy_score(Y_test,y_pred)
print(score)
filename = 'Drowsiness_model.h5'
pickle.dump(logreg, open(filename, 'wb'))
