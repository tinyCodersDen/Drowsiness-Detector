# from sklearn import svm
# import pandas as pd
# import mediapipe as mp
# df = pd.read_csv("Drowsy.csv")
# X = []
# y = []
# for index, row in df.iterrows():
#     l = [row['23'], row['24'],row['25'],row['26'],row['27'],row['28'],row['29'],row['30'],row['31'],row['253'],row['254'],row['255'],row['256'],row['257'],row['258'],row['259'],row['260'],row['261'],row['185'],row['40'],row['39'],row['0'],row['270'],row['269'],row['409'],row['375'],row['321'],row['405'],row['314'],row['17'],row['84'],row['181'],row['91'],row['146']]
#     X.append(l)
#     y.append(row['Status'])
# clf = svm.SVC()
# clf.fit(X, y)
# print(clf.predict([[0,0]]))
##############################
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
# logreg.fit(l, y)
# y_pred = logreg.predict(X_test)
# score = accuracy_score(Y_test,y_pred)
# print(score)
# filename = 'Drowsiness_model.h5'
# pickle.dump(logreg, open(filename, 'wb'))