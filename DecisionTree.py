import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = {
    "Hours":       [2.0, 3.5, 5.0, 7.0, 1.0, 6.0, 4.5, 1.5],
    "Assignments": [1,   2,   3,   4,   1,   4,   2,   0],
    "Attendance":  [60,  65,  75,  85,  50,  90,  70,  40],
    "Passed":      [0,   0,   1,   1,   0,   1,   1,   0]
}

df = pd.DataFrame(data)
x = df[["Hours","Assignments","Attendance"]]
y = df["Passed"]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(f"{acc*100:.2f}%")


plt.figure(figsize=(10,6))
plot_tree(model, feature_names=["Hours", "Assignments", "Attendance"], class_names=["Fail", "Pass"], filled=True)
plt.title("Decision Tree for Passing Prediction")
plt.show()