# ============================================
# PHASE 1 — BASIC F1 PREDICTION MODEL
# ============================================

# --------------------------------------------
# 1. IMPORT LIBRARIES
# --------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# --------------------------------------------
# 2. CREATE SAMPLE F1 DATASET
# --------------------------------------------

data = {
    "Driver": [
        "Verstappen",
        "Norris",
        "Leclerc",
        "Hamilton",
        "Russell",
        "Piastri",
        "Sainz",
        "Perez",
        "Alonso",
        "Gasly"
    ],

    "Grid": [1, 3, 2, 5, 4, 6, 7, 8, 9, 10],

    "Rain": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],

    "Tire": [
        "Soft",
        "Medium",
        "Soft",
        "Hard",
        "Medium",
        "Soft",
        "Hard",
        "Medium",
        "Soft",
        "Hard"
    ],

    "Position": [1, 2, 3, 5, 4, 6, 8, 7, 9, 10]
}

df = pd.DataFrame(data)


# --------------------------------------------
# 3. DISPLAY DATASET
# --------------------------------------------

print("\n===== ORIGINAL DATASET =====\n")
print(df)


# --------------------------------------------
# 4. CHECK FOR MISSING VALUES
# --------------------------------------------

print("\n===== MISSING VALUES =====\n")
print(df.isnull().sum())


# --------------------------------------------
# 5. ENCODE CATEGORICAL DATA
# --------------------------------------------

# ML models cannot understand text directly

driver_encoder = LabelEncoder()
tire_encoder = LabelEncoder()

df["Driver"] = driver_encoder.fit_transform(df["Driver"])
df["Tire"] = tire_encoder.fit_transform(df["Tire"])


print("\n===== ENCODED DATASET =====\n")
print(df)


# --------------------------------------------
# 6. FEATURE SELECTION
# --------------------------------------------

# X = INPUT FEATURES
X = df[["Driver", "Grid", "Rain", "Tire"]]

# y = TARGET VARIABLE
y = df["Position"]


# --------------------------------------------
# 7. SCALE FEATURES
# --------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n===== SCALED FEATURES =====\n")
print(X_scaled)


# --------------------------------------------
# 8. SPLIT DATASET
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== TRAINING DATA SIZE =====")
print(len(X_train))

print("\n===== TESTING DATA SIZE =====")
print(len(X_test))


# --------------------------------------------
# 9. CREATE MODEL
# --------------------------------------------

model = LinearRegression()


# --------------------------------------------
# 10. TRAIN MODEL
# --------------------------------------------

model.fit(X_train, y_train)

print("\n===== MODEL TRAINED SUCCESSFULLY =====")


# --------------------------------------------
# 11. MAKE PREDICTIONS
# --------------------------------------------

predictions = model.predict(X_test)

print("\n===== PREDICTIONS =====\n")
print(predictions)


# --------------------------------------------
# 12. COMPARE ACTUAL VS PREDICTED
# --------------------------------------------

comparison = pd.DataFrame({
    "Actual Position": y_test.values,
    "Predicted Position": predictions
})

print("\n===== ACTUAL VS PREDICTED =====\n")
print(comparison)


# --------------------------------------------
# 13. EVALUATE MODEL
# --------------------------------------------

mae = mean_absolute_error(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print(f"Mean Absolute Error: {mae}")


# --------------------------------------------
# 14. CORRELATION MATRIX
# --------------------------------------------

print("\n===== CORRELATION MATRIX =====\n")
print(df.corr(numeric_only=True))


# --------------------------------------------
# 15. VISUALIZE GRID VS POSITION
# --------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(df["Grid"], df["Position"])

plt.xlabel("Starting Grid Position")
plt.ylabel("Final Race Position")
plt.title("Grid Position vs Final Position")

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.show()


# --------------------------------------------
# 16. VISUALIZE PREDICTION ERRORS
# --------------------------------------------

errors = y_test.values - predictions

plt.figure(figsize=(8, 5))

plt.bar(range(len(errors)), errors)

plt.xlabel("Test Sample")
plt.ylabel("Prediction Error")
plt.title("Prediction Errors")

plt.show()


# --------------------------------------------
# 17. TEST CUSTOM INPUT
# --------------------------------------------

# Example:
# Driver = Verstappen
# Grid = 1
# Rain = 0
# Tire = Soft

custom_data = pd.DataFrame({
    "Driver": ["Verstappen"],
    "Grid": [1],
    "Rain": [0],
    "Tire": ["Soft"]
})


# Encode custom input
custom_data["Driver"] = driver_encoder.transform(custom_data["Driver"])
custom_data["Tire"] = tire_encoder.transform(custom_data["Tire"])


# Scale custom input
custom_scaled = scaler.transform(custom_data)


# Predict
custom_prediction = model.predict(custom_scaled)

print("\n===== CUSTOM PREDICTION =====")
print(f"Predicted Final Position: {custom_prediction[0]:.2f}")


