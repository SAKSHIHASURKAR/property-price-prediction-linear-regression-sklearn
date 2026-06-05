## Property Price Prediction Using Multiple Linear Regression
A machine learning project that predicts residential property prices using Multiple Linear Regression and compares the impact of different feature combinations on prediction accuracy.
## Overview
Property price prediction is one of the most practical applications of machine learning in the real estate industry. House prices are influenced by several factors such as property size, number of bedrooms, number of bathrooms, lot size, and construction year.
This project uses Multiple Linear Regression to analyse how different combinations of property features affect prediction performance. Five different regression models were trained using different subsets of features to systematically evaluate the contribution of each variable to house price prediction.

The project focuses on:                        

•	Predicting residential property prices  
•	Evaluating the impact of different property features          
•	Comparing multiple feature combinations  
•	Analysing model performance using statistical metrics   
•	Visualising model results through comparative charts    

The analysis provides valuable insights into which property characteristics have the greatest influence on market value.


## Problem Statement 

Real estate pricing depends on multiple interconnected variables, making accurate prediction a challenging task.

This project aims to answer the following questions:

•	Which feature combination produces the best prediction accuracy?    
•	Which individual feature has the greatest impact on house prices?   
•	How does removing specific features affect model performance?  
•	Which variables contribute most to explaining property value?  

The objective is to build and evaluate Multiple Linear Regression models that can accurately estimate residential property prices while identifying the most influential predictors.

## Dataset  
Dataset Name

property.csv

Dataset Description

The dataset contains residential property listings with structural and physical attributes used to predict market prices.

Dataset Features

| Feature          | Description                       |
| ---------------- | --------------------------------- |
| `Square_Footage` | Total area of the property        |
| `Num_Bedrooms`   | Number of bedrooms                |
| `Num_Bathrooms`  | Number of bathrooms               |
| `Year_Built`     | Construction year of the property |
| `Lot_Size`       | Total lot size                    |
| `Price`          | Target variable (property price)  |

## Tools and Technologies 

| Tool / Technology | Purpose                                        |
| ----------------- | ---------------------------------------------- |
| **Python 3**      | Programming language                           |
| **Google Colab**  | Development environment                        |
| **Pandas**        | Data loading and manipulation                  |
| **Scikit-learn**  | Machine learning model training and evaluation |
| **Matplotlib**    | Data visualisation                             |
| **Seaborn**       | Statistical plotting                           |

## Methods 
The project follows a systematic machine learning workflow:

1. Data Loading

•	Loaded the property dataset using pandas 

•	Inspected the dataset structure and columns

    df = pd.read_csv("property (1).csv")

2. Feature Engineering

Five different feature combinations were created to analyse the importance of each variable.
| Model   | Features Used           |
| ------- | ----------------------- |
| Model 1 | All 5 features          |
| Model 2 | Without `Lot_Size`      |
| Model 3 | Without `Year_Built`    |
| Model 4 | Without `Num_Bathrooms` |
| Model 5 | Without `Num_Bedrooms`  |

This systematic feature elimination approach helps identify the importance of individual features.

3. Train-Test Split

The dataset was split into:

80% Training Data

20% Testing Data

    train_test_split(test_size=0.2, random_state=42)

4. Model Training 

A separate Multiple Linear Regression model was trained for each feature combination using Scikit-learn's LinearRegression algorithm.

5. Model Evaluation 

The following performance metrics were calculated:   
•	Mean Squared Error (MSE)    
•	Training R² Score    
•	Testing R² Score  

6. Comparison and Visualisation    

Results from all models were compared using:    
•	Test R² comparison charts   
•	Mean Squared Error comparison charts  
•	Model ranking tables
## Key Insights

The model containing all five features achieved the highest predictive performance with a Test R² score of approximately 0.994.  

1. Number of Bathrooms is Highly Influential   

Removing Num_Bathrooms caused the largest reduction in prediction accuracy, indicating that it is the most important predictor in the dataset.  

2. Strong Linear Relationships  

All models achieved high R² values, suggesting strong linear relationships between property attributes and house prices.   

3. Feature Selection Matters

Even removing a single feature resulted in measurable performance degradation, demonstrating the importance of feature selection in predictive modelling.




               
## Dashboard / Output / Model

The project generates several analytical visualisations:

| # | Visualisation | Description |
|---|--------------|-------------|
| 1 | Test R² by Model | Compares predictive performance across different feature combinations |
| 2 | MSE by Model | Compares prediction errors across all regression models |
| 3 | Model Performance Summary Table | Displays the number of features, MSE, Train R², and Test R² for each model |
| 4 | Feature Combination Comparison | Evaluates the impact of removing specific features on prediction accuracy |

### Output Generated

| Output | Description |
|---------|------------|
| Model Performance Comparison Table | Summary of all model evaluation metrics |
| Test R² Visualisation | Bar chart showing predictive accuracy of each model |
| Mean Squared Error Visualisation | Bar chart comparing prediction errors |
| Feature Importance Insights | Analysis of how feature removal affects model performance |
| Model Ranking Summary | Models ranked according to Test R² performance |
## How to Run This Project 
Option 1 — Local Environment  

Step 1 — Clone the Repository  

git clone https://github.com/<your-username>/property-price-prediction.git
cd property-price-prediction
________________________________________
Step 2 — Create Virtual Environment

python -m venv venv
Activate it:
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
________________________________________
Step 3 — Install Dependencies

pip install -r requirements.txt
________________________________________
Step 4 — Add Dataset

Place the dataset inside the data/ folder:   
project-root/   
│── data/  
│    └── property.csv
________________________________________
Step 5 — Run the Notebook

Launch Jupyter Notebook:
jupyter notebook
Open:
 Property Price Prediction.ipynb
Run all cells sequentially.
________________________________________
Step 6 — View Outputs 

After execution, results will include:  
•	Model performance metrics  
•	Test R² comparison chart  
•	MSE comparison chart  
•	Results summary table

________________________________________ 

Project Structure
property-price-prediction/            
│── data/              
│    └── property.csv        
│
│── notebooks/        
│    └── Property Price Prediction.ipynb    
│
│── outputs/       
│    ├── plots/       
│    └── results/    
│
│── requirements.txt    
│── README.md


Notes   
•	Run all notebook cells sequentially.  
•	Ensure the dataset is placed in the correct directory.   
•	Install all dependencies before execution.  
•	Google Colab can be used without local installation. 

## Results and Conclusion  
The analysis demonstrates that Multiple Linear Regression is highly effective for residential property price prediction. 

Major Findings  
•	The full feature model achieved the highest accuracy.  
•	Number of bathrooms emerged as the most influential predictor.  
•	All models showed strong predictive performance.  
•	Feature elimination revealed the contribution of individual variables.  
•	R² and MSE provided effective measures for comparing model quality.  

Conclusion  
The project successfully demonstrates how Multiple Linear Regression can be used to predict residential property prices and evaluate feature importance.
The results show that including all relevant property attributes produces the most accurate predictions, while feature selection significantly influences model performance.


## Future Work  

Possible future improvements include: 

•	Feature scaling using StandardScaler  
•	Ridge Regression implementation   
•	Lasso Regression implementation  
•	Cross-validation for robust evaluation  
•	Correlation heatmap analysis  
•	Random Forest Regression  
•	Decision Tree Regression  
•	Gradient Boosting Models  
•	Inclusion of location-based property features  

## Author and Contact 

Author

Sakshi Ashok Hasurkar   
AI & Data Analytics Enthusiast | Master's Student in Artificial Intelligence

I am a Computer Engineer with experience in software testing and a strong interest in data analytics and machine learning. Currently pursuing a Master's in Artificial Intelligence at the University of East London, I am focused on building data-driven solutions and predictive analytics models using Python and Machine Learning techniques.  
•	Master's in Artificial Intelligence – University of East London  
•	1+ year experience as Software Tester (NHS UK Project)  
•	Skills: Python, Machine Learning, Data Analysis, Statistics, Data Visualisation  
•	London, UK  

Connect with Me  
•	LinkedIn: www.linkedin.com/in/sakshi-hasurkar-57b2412bb
