import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = {
    "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 
              7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 
              2.7, 4.8, 3.8, 6.9, 7.8],
    "Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 
               85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 
               30, 54, 35, 76, 86]
}

df = pd.DataFrame(data)

x = df[["Hours"]]
y = df["Scores"]
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_pred,y_test)
print(f"error = {mse:.2f}")

plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', label='Predicted Line')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Test Data vs Prediction")
plt.legend()
plt.grid(True)
plt.show()