import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
data = {
    "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 
              7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 
              2.7, 4.8, 3.8, 6.9, 7.8],
    "Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 
               85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 
               30, 54, 35, 76, 86]
}

df = pd.DataFrame(data)

# Splitting input and output
x = df[["Hours"]]
y = df["Scores"]

# Model
model = LinearRegression()
model.fit(x, y)

# Prediction for 4.2 hours
hours = 4.2
predicted_score = model.predict([[hours]])
print(f"If you study {hours} hours, you will score {predicted_score[0]:.2f} marks")

# Plot original data
plt.scatter(x, y, color='blue', label='Actual data')

# Plot regression line
plt.plot(x, model.predict(x), color='red', label='Best-fit line')

# Plot predicted point
plt.scatter([hours], predicted_score, color='green', s=100, label='Predicted score')  # big green dot

# Labels and formatting
plt.title('Hours vs Scores (with Prediction)')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()

