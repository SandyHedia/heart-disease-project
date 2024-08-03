# Heart Disease Prediction Project

## Project Overview
This project aims to predict the presence of heart disease in patients using various machine learning models. Two separate implementations are provided: one using scikit-learn and another using PyTorch. The project involves data preprocessing, feature extraction, model training, evaluation, and deployment.

## Datasets
The dataset used for this project is the [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) from the UCI Machine Learning Repository. It includes several features such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, oldpeak, the slope of the peak exercise ST segment, number of major vessels, and thalassemia.

## Project Structure
The project is divided into two main parts:
1. **Scikit-learn Implementation**
2. **PyTorch Implementation**

### 1. Scikit-learn Implementation
- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling.
- **Model Training**: Training several classifiers including Logistic Regression, Random Forest, SVM, Gradient Boosting, and k-Nearest Neighbors.
- **Hyperparameter Tuning**: Using GridSearchCV for hyperparameter optimization.
- **Model Evaluation**: Evaluating models using accuracy, precision, recall, F1-score, and ROC-AUC.
- **Feature Importance Analysis**: Analyzing the importance of features using model-specific techniques.

### 2. PyTorch Implementation
- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling.
- **Model Architecture**: Building a feedforward neural network (FNN).
- **Training**: Training the FNN using PyTorch with various hyperparameters.
- **Evaluation**: Evaluating the neural network using accuracy and loss curves.
- **Visualization**: Plotting training loss and accuracy over epochs.

## Results
- **Scikit-learn Implementation**: Achieved a final model accuracy of 98% using a Random Forest classifier.
- **PyTorch Implementation**: Achieved a comparable accuracy using a feedforward neural network.

## Requirements
- Python 3.6+
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, PyTorch, Jupyter Notebook

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks**:
   Open the notebooks using Jupyter Notebook or Jupyter Lab and run all cells to see the results.
   - `predicting_heart_disease_using_sklearn_.ipynb`
   - `predicting_heart_diseases_project_using_pytorch.ipynb`

## Conclusion
This project demonstrates the application of machine learning techniques to predict heart disease, showcasing both traditional machine learning models and neural networks. The results highlight the importance of feature engineering and model selection in achieving high accuracy.

## Contact
For any questions or suggestions, please contact Sandy at [sandyhedia@gmail.com].

