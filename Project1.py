import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = {"Hours" : [1.0, 3.5, 5.0, 7.0, 2.0, 6.0, 4.5, 1.5],
        "Passed" : [0,   1,   1,   1,   1,   0,   1,   1]}

df = pd.DataFrame(data)

x = df[["Hours"]]
y = df["Passed"]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)
model = LogisticRegression()
model.fit(x_train,y_train)
pass_pred = model.predict(x_test)
msd = accuracy_score(y_test,pass_pred)
print(f"{msd*100:.2f}")

plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, pass_pred, color='red', label='Predicted Line')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Test Data vs Prediction")
plt.legend()
plt.grid(True)
plt.show()