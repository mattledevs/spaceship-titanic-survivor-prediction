# Spaceship Titanic Survivor Prediction

Kaggle project on predicting whether passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

## Problem

In this Kaggle competition, the task is to predict which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly. The dataset contains personal records recovered from the ship's damaged computer system, including passenger information such as age, home planet, destination, cabin details, and spending records.

The challenge involves binary classification where the target variable is "Transported" (True/False). The dataset has missing values, categorical features, and potential class imbalance that need to be addressed for optimal model performance.

## Approach

The solution follows a comprehensive machine learning pipeline:

1. **Data Exploration and Cleaning**: Handle missing values, detect duplicates, and perform initial data analysis.
2. **Feature Engineering**: Parse cabin information, create group features, and derive expenditure-related features.
3. **Outlier Detection and Treatment**: Identify and cap outliers in expenditure data.
4. **Data Consistency Checks**: Ensure logical consistency across features.
5. **Encoding and Scaling**: Transform categorical variables and scale numerical features.
6. **Model Development**: Compare multiple algorithms, perform hyperparameter tuning, and address class imbalance.
7. **Advanced Techniques**: Implement SMOTE for class balancing and additional feature engineering.

## Methodology

### Data Loading and Exploration
- Loaded training and test datasets (`train.csv`, `test.csv`)
- Performed initial data exploration including shape, data types, and missing value analysis
- Visualized data distributions and missing patterns

### Data Cleaning
- **Missing Values**: Imputed categorical features with mode, numerical features with median, and expenditures with 0
- **Duplicates**: Verified no duplicate rows in either dataset
- **Cabin Parsing**: Split cabin strings into Deck, Cabin_Num, and Side components
- **PassengerId Parsing**: Extracted Group and Person numbers, calculated group sizes

### Feature Engineering
- **Expenditure Features**: Created Total_Expenditure, Luxury_Expenditure, Basic_Expenditure, and Has_Spent indicators
- **Age Groups**: Categorized ages into Child, Teen, Young_Adult, Adult, and Senior
- **Advanced Features**: Added interaction features like Age_Expenditure_Interaction, Group_VIP_Interaction, CryoSleep_Expenditure, Age_CryoSleep_Interaction, and Luxury_Basic_Ratio

### Outlier Treatment
- Detected outliers in expenditure columns using IQR method
- Capped extreme values to reduce their impact on model training
- Ensured logical consistency (e.g., CryoSleep passengers have zero expenditures)

### Data Consistency Checks
- Verified cabin numbering patterns
- Checked group consistency for HomePlanet and Destination
- Validated engineered feature calculations

### Preprocessing
- **Encoding**: One-hot encoded categorical features (HomePlanet, Destination, Deck, Side, Age_Group)
- **Scaling**: Standardized numerical features using StandardScaler
- **Boolean Handling**: Converted boolean columns appropriately

### Model Development
- **Baseline Model**: Logistic Regression for initial performance benchmark
- **Model Comparison**: Evaluated Logistic Regression, Random Forest, and XGBoost
- **Hyperparameter Tuning**: Grid search optimization for XGBoost parameters
- **Class Imbalance**: Applied SMOTE oversampling to balance the target classes
- **Advanced Feature Engineering**: Added interaction features for improved predictive power

### Evaluation Metrics
- Primary metric: F1 Macro Average (harmonic mean of precision and recall for both classes)
- Secondary metrics: Accuracy, AUC-ROC, F1 Weighted Average

## Results

### Model Performance Comparison

| Model | Accuracy | AUC-ROC | F1 Macro |
|-------|----------|---------|----------|
| Logistic Regression (Baseline) | 0.7894 | 0.8678 | 0.7889 |
| Random Forest | 0.8055 | 0.8814 | 0.8053 |
| XGBoost (Tuned) | 0.8108 | 0.8872 | 0.8107 |
| XGBoost + SMOTE | 0.8108 | 0.8872 | 0.8107 |
| XGBoost + Advanced Features | 0.8108 | 0.8872 | 0.8107 |

### Final Model Details
- **Best Model**: Tuned XGBoost with advanced feature engineering
- **Validation F1 Macro**: 0.8107
- **Validation Accuracy**: 0.8108
- **Validation AUC-ROC**: 0.8872
- **Total Features**: 37 (32 original + 5 interaction features)

### Key Insights
- XGBoost outperformed other algorithms with proper tuning
- Advanced feature engineering provided marginal improvements over baseline
- SMOTE helped address class imbalance but didn't significantly boost F1 scores
- Top predictive features included CryoSleep status, expenditure patterns, and cabin deck information

### Submission
The final model predictions were saved to `submission.csv` for Kaggle submission, achieving a competitive score on the leaderboard.

## Dependencies

- Python 3.11
- pandas, numpy
- scikit-learn
- xgboost
- imbalanced-learn (for SMOTE)
- matplotlib, seaborn (for visualization)

## Usage

1. Ensure all dependencies are installed
2. Run the Jupyter notebook `spaceship-titanic-survivors.ipynb`
3. Execute cells sequentially to reproduce the analysis
4. The final submission file will be generated as `submission.csv`
