# 🏎️ F1 Prediction Model — Phase 1

A beginner-friendly Machine Learning project that predicts Formula 1 race finishing positions using basic race data.

This project is built to learn:

* Data Science fundamentals
* Machine Learning workflow
* Feature engineering
* Model evaluation
* Python ML libraries

---

# 📌 Project Goal

The goal of this project is to build a simple AI model capable of predicting an F1 driver's finishing position based on:

* Starting grid position
* Weather conditions
* Tire compound
* Driver information

---

# 🧠 Concepts Learned

This phase covers the core foundations of Machine Learning:

* Datasets
* Features & Targets
* Data Cleaning
* Encoding
* Feature Scaling
* Train/Test Split
* Linear Regression
* Predictions
* Mean Absolute Error (MAE)
* Correlation Analysis
* Data Visualization

---

# 📦 Libraries Used

Install required packages:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
```

---

# 📂 Project Structure

```text
F1-Prediction-Model/
│
├── model.ipynb
├── README.md
```

---

# ⚙️ Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming Language |
| Pandas       | Data Handling        |
| NumPy        | Numerical Operations |
| Matplotlib   | Data Visualization   |
| Scikit-Learn | Machine Learning     |

---

# 📊 Dataset Used

A small manually created dataset containing:

| Driver     | Grid | Rain | Tire   | Position |
| ---------- | ---- | ---- | ------ | -------- |
| Verstappen | 1    | 0    | Soft   | 1        |
| Norris     | 3    | 1    | Medium | 2        |

---

# 🔍 Machine Learning Workflow

```text
Collect Data
     ↓
Clean Data
     ↓
Encode Categorical Data
     ↓
Scale Features
     ↓
Split Dataset
     ↓
Train Model
     ↓
Make Predictions
     ↓
Evaluate Performance
```

---

# 🤖 Model Used

## Linear Regression

The model learns relationships between:

* Starting position
* Weather
* Tire strategy
* Driver data

and predicts the final finishing position.

---

# 📈 Visualizations Included

The notebook contains:

* Grid Position vs Final Position graph
* Prediction Error graph
* Correlation Matrix

---

# 📉 Evaluation Metric

## Mean Absolute Error (MAE)

Measures how far predictions are from actual results.

Lower MAE = Better model.

---

# 🚀 How To Run

## 1. Clone Repository

```bash
git clone https://github.com/lxcky9/F1.git
```

---

## 2. Open Project Folder

```bash
cd F1-Prediction-Model
```

---

## 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
model.ipynb
```

Run all cells.

---

# 🏁 Example Prediction

Input:

```python
Driver = Verstappen
Grid = 1
Rain = 0
Tire = Soft
```

Output:

```text
Predicted Final Position: 1.8
```

---

# 📚 Learning Outcome

After completing this phase you will understand:

* Basic Machine Learning workflow
* Data preprocessing
* Training and testing models
* Prediction systems
* Performance evaluation

---

# 🔮 Future Improvements

Upcoming phases will include:

* Real F1 datasets
* FastF1 API integration
* XGBoost models
* Random Forest
* LSTM Neural Networks
* Telemetry analysis
* Weather integration
* Live prediction dashboard

---

# 👨‍💻 Author

Lavanya Patil
B.Tech CSBS Student
Aspiring Data Scientist & ML Engineer
