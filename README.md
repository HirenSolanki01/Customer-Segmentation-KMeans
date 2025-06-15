# 🧠 Customer Segmentation using K-Means Clustering

This project applies the K-Means clustering algorithm to group retail store customers based on their age, annual income, and spending score. The goal is to identify meaningful customer segments for better marketing and product targeting.

## 📂 Dataset
- Dataset: Mall_Customers.csv
- Source: [Kaggle - Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## 📊 Features Used
- Age
- Annual Income (k$)
- Spending Score (1–100)

## ⚙️ Techniques Applied
- Data Cleaning & Feature Selection
- Standardization with `StandardScaler`
- Elbow Method to find optimal number of clusters (K)
- K-Means Clustering from `sklearn`
- Cluster Visualization using `matplotlib` and `seaborn`

## 📌 Result
Customers were grouped into distinct clusters like:
- High spenders with high income
- Low income, low spending
- Young average spenders, etc.

## 🧪 Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

## 📉 Visual Output
- Elbow Curve
- 2D Cluster Scatter Plot

## 🚀 Usage
You can use this project to:
- Understand unsupervised learning (clustering)
- Build customer personas in marketing
- Apply K-Means to other segmentation problems

Created as part of a machine learning practice project to understand clustering and customer analytics.

