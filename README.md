📌 Project Overview

This repository contains an end-to-end machine learning pipeline built using Snowflake Feature Store and Python (scikit-learn).
The project explores the Titanic dataset and demonstrates how to:

Prepare and explore raw data in Snowflake ❄️

Engineer meaningful features (age groups, family size, cabin indicator, etc.) 🛠️

Store engineered features in a Feature Store for ML pipelines 📦

Train and evaluate a Logistic Regression model to predict passenger survival 🤖

Visualize results with accuracy scores, predictions, and confusion matrices 📊

⚙️ Workflow

1️⃣ Data Extraction & Exploration

Create Snowflake database, schema, and warehouse.

Profile Titanic dataset for missing values, distributions, and statistics.

2️⃣ Feature Engineering

Handle missing values (median age, median fare, etc.).

Encode categorical variables (sex, embarked, cabin).

Derive new features like family size, age groups, and title extraction.

3️⃣ Feature Store

Save the engineered dataset to FEATURE_STORE.TITANIC_FEATURES_FINAL.

Ensure features are reusable and consistent across ML pipelines.

4️⃣ Model Training (Python)

Load features with Snowpark into Pandas.

Split into train (70%) and test (30%).

Train Logistic Regression using scikit-learn.

5️⃣ Evaluation

Compute model accuracy (~82%).

Inspect sample predictions (Actual vs Predicted).

Generate a confusion matrix heatmap.
