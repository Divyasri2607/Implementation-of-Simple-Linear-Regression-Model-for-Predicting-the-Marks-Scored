# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard Libraries.

2.Set variables for assigning dataset values and import linear regression from sklearn.

3.Assign the points for representing in the graph.

4.Predict the regression for marks by using the representation of the graph and compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Divya Sri V
RegisterNumber: 212224230070

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(*X)
Y=df.iloc[:,1].values
print(*Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
print(*Y_pred)
Y_test
print(*Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse) 
```

## Output:
<img width="977" height="831" alt="Screenshot 2025-09-03 134927" src="https://github.com/user-attachments/assets/130ef805-3386-448a-8769-de6185beaa3b" />
<img width="1735" height="822" alt="Screenshot 2025-09-03 135025" src="https://github.com/user-attachments/assets/e6f3ca0d-baab-4218-b166-3dae89fc2bdb" />
<img width="936" height="856" alt="Screenshot 2025-09-03 135106" src="https://github.com/user-attachments/assets/e8f19619-b905-4d61-8996-e874ca9b8316" />
<img width="935" height="841" alt="Screenshot 2025-09-03 135142" src="https://github.com/user-attachments/assets/ab6c8c71-4c4b-4bc6-8c35-7d1c15cf9765" />
<img width="681" height="265" alt="Screenshot 2025-09-03 135216" src="https://github.com/user-attachments/assets/c2a7b8aa-81d1-42d6-a69d-61b22aca5337" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
