# Ex.No: 6               HOLT WINTERS METHOD
### Date: 19-05-2026
### RegNo: 212223240082

### AIM:
To implement and evaluate the Holt-Winters Exponential Smoothing method for time series forecasting using the Air Passengers dataset.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# Load IPL dataset
data = pd.read_csv('matches.csv')

# Convert date column
data['date'] = pd.to_datetime(
    data['date'],
    format='mixed',
    dayfirst=True
)

# Set date as index
data.set_index('date', inplace=True)

# Monthly IPL matches count
data_monthly = data['id'].resample('MS').count()

# Convert to DataFrame
data_monthly = pd.DataFrame(data_monthly)
data_monthly.columns = ['Matches']

# Display first rows
display(data_monthly.head())

# Plot original data
data_monthly.plot(figsize=(12, 6))

plt.title('Monthly IPL Matches Data')
plt.xlabel('Date')
plt.ylabel('Number of Matches')
plt.grid(True)
plt.show()

# Scaling
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        data_monthly.values.reshape(-1, 1)
    ).flatten()
)

# Plot scaled data
scaled_data.plot(figsize=(12, 6))

plt.title('Scaled Monthly IPL Matches Data')
plt.xlabel('Time')
plt.ylabel('Scaled Matches')
plt.grid(True)
plt.show()

# Seasonal decomposition
decomposition = seasonal_decompose(
    data_monthly,
    model="additive",
    period=12
)

decomposition.plot()
plt.show()

# Avoid zeros for multiplicative model
scaled_data = scaled_data + 1

# Train-Test Split
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Holt-Winters Model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast on test data
test_predictions_add = model_add.forecast(
    steps=len(test_data)
)

# Plot predictions
plt.figure(figsize=(12, 6))

ax = train_data.plot(label='Train Data')

test_data.plot(ax=ax, label='Test Data')

test_predictions_add.plot(
    ax=ax,
    label='Test Predictions'
)

ax.legend()

ax.set_title(
    'Holt-Winters Predictions vs Actual IPL Matches'
)

plt.xlabel('Time')
plt.ylabel('Scaled Matches')

plt.grid(True)
plt.show()

# Evaluation Metrics
rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_add
    )
)

mae = mean_absolute_error(
    test_data,
    test_predictions_add
)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

print(
    f"Scaled data variance: {np.sqrt(scaled_data.var()):.4f}, "
    f"Scaled data mean: {scaled_data.mean():.4f}"
)

# Final model using full data

final_model = ExponentialSmoothing(
    data_monthly['Matches'],
    trend='add',
    seasonal='add',   # changed from 'mul' to 'add'
    seasonal_periods=12
).fit()

# Forecast next 12 months
final_predictions = final_model.forecast(steps=12)

# Plot future predictions
plt.figure(figsize=(12, 6))

ax = data_monthly['Matches'].plot(
    label='Historical Data'
)

final_predictions.plot(
    ax=ax,
    label='Future Predictions'
)

ax.legend()

ax.set_xlabel('Months')
ax.set_ylabel('Number of IPL Matches')

ax.set_title('Prediction: Monthly IPL Matches')

plt.grid(True)
plt.show()
```

### OUTPUT:

TEST_PREDICTION
<img width="1121" height="612" alt="image" src="https://github.com/user-attachments/assets/1a3c0533-8bd1-4d22-9f5a-c96f8b37a916" />

FINAL_PREDICTION
<img width="1126" height="668" alt="image" src="https://github.com/user-attachments/assets/a0b62368-fa53-4fe5-8422-f51063ddc890" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
