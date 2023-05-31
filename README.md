# CompasScore
## COMPAS Score Raw Dataset

The COMPAS Score Raw dataset contains information about the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) system used in the criminal justice system to assess the risk of recidivism among defendants. The data includes demographic characteristics such as age, sex, ethnic origin, as well as the COMPAS scores assigned to the defendants.

### Objective
The objective of this dataset is to examine the fairness and potential biases of the COMPAS system. By using analysis techniques, we can evaluate how factors such as ethnic origin and sex influence the COMPAS scores assigned to defendants.

### Using SHAP for Model Analysis
SHAP (SHapley Additive exPlanations) is an analysis technique based on game theory that allows us to understand the importance of different features in the predictions of machine learning models. By applying SHAP to models built on the COMPAS Score Raw dataset, we can uncover the factors that influence the model's predictions, including potential biases or discriminatory patterns.

### Analysis Steps
1. Data preprocessing: Clean the data, handle missing values...
2. Data splitting: Split the data into training and test sets.
3. Model training: Use machine learning algorithms, such as gradient boosting classifiers and XGBoost, to train models on the training set.
4. Application of SHAP: Use the SHAP library to analyze the trained models and obtain explanations for the predictions.
5. Interpretation of SHAP results: Analyze the contributions of each feature, particularly ethnic origin and sex, to understand how they influence the model's decisions and detect potential biases.
6. Action: Develop strategies to mitigate biases and improve the fairness of the model.

### References
* COMPAS Score Raw dataset: https://www.kaggle.com/datasets/danofer/compass
* SHAP (SHapley Additive exPlanations): https://shap.readthedocs.io/en/latest/index.html
