
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="H")

data = pd.DataFrame({
    "timestamp": dates,
    "power_output": np.random.uniform(300, 1000, len(dates)),  # in Watts
    "voltage": np.random.uniform(20, 40, len(dates)),
    "current": np.random.uniform(5, 15, len(dates)),
    "temperature": np.random.uniform(25, 60, len(dates)),
    "air_quality": np.random.uniform(10, 300, len(dates)),  # Dust level
    "humidity": np.random.uniform(20, 90, len(dates)),
    "wind_speed": np.random.uniform(0, 15, len(dates)),
    "rainfall": np.random.uniform(0, 5, len(dates))
})


data["efficiency"] = (
    (data["power_output"] / data["power_output"].max()) * 100
    - (data["air_quality"] / 300) * 20
    - (data["temperature"] - 25) * 0.1
)


data["last_cleaned_days"] = np.random.randint(0, 10, len(data))
data["needs_cleaning"] = np.where(data["last_cleaned_days"] > 7, 1, 0)

print("✅ Data Loaded Successfully!")
print(data.head())



data["hour"] = data["timestamp"].dt.hour
data["day"] = data["timestamp"].dt.day
data["month"] = data["timestamp"].dt.month


features = ["voltage", "current", "temperature", "air_quality", "humidity",
            "wind_speed", "rainfall", "hour", "day", "month"]

X = data[features]
y_eff = data["efficiency"]                
y_clean = data["needs_cleaning"]         

# Split data
X_train, X_test, y_eff_train, y_eff_test = train_test_split(X, y_eff, test_size=0.2, random_state=42)
_, _, y_clean_train, y_clean_test = train_test_split(X, y_clean, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



regressor = RandomForestRegressor(n_estimators=150, random_state=42)
regressor.fit(X_train_scaled, y_eff_train)

y_eff_pred = regressor.predict(X_test_scaled)

print("\n📊 Efficiency Prediction Model Performance:")
print("MAE:", round(mean_absolute_error(y_eff_test, y_eff_pred), 2))
print("R² Score:", round(r2_score(y_eff_test, y_eff_pred), 3))



classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_clean_train)

y_clean_pred = classifier.predict(X_test_scaled)

print("\n🧹 Cleaning Prediction Model Performance:")
print("Accuracy:", round(accuracy_score(y_clean_test, y_clean_pred) * 100, 2), "%")
print("\n", classification_report(y_clean_test, y_clean_pred))



avg_cleaning_interval_days = int(data[data["needs_cleaning"] == 1]["last_cleaned_days"].mean())
next_cleaning_date = datetime.now() + timedelta(days=avg_cleaning_interval_days)

print(f"\n🗓️ Estimated Next Cleaning Date: {next_cleaning_date.strftime('%Y-%m-%d')}")



plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.lineplot(x=data["timestamp"], y=data["efficiency"], label="Efficiency (%)")
plt.title("Solar Panel Efficiency Over Time")
plt.xlabel("Date")
plt.ylabel("Efficiency (%)")
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=data["temperature"], y=data["efficiency"], hue=data["air_quality"], palette="coolwarm")
plt.title("Efficiency vs Temperature & Dust Level")
plt.xlabel("Temperature (°C)")
plt.ylabel("Efficiency (%)")
plt.tight_layout()
plt.show()



sample_input = np.array([[30, 10, 45, 150, 60, 5, 0, 14, 25, 3]])  # Example values
sample_input_scaled = scaler.transform(sample_input)

pred_eff = regressor.predict(sample_input_scaled)[0]
pred_clean = classifier.predict(sample_input_scaled)[0]

print("\n⚙️ Real-time Prediction Example:")
print(f"Predicted Efficiency: {pred_eff:.2f}%")
print("Maintenance Required:" , "Yes" if pred_clean == 1 else "No")

print("\n✅ AI System Completed Successfully!")
