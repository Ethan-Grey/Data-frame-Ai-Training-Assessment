# 📦 Imports
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from User_Input_GUI import launch_multi_entry_gui

# 📊 Model Imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Task 1: Import the dataset
def load_data(): # function called load_data
    data = pd.read_excel("Net_Worth_Data.xlsx") # reads the excel file
    return data # returns the data

data = load_data()

# Task 2: Display first 5 rows of the dataset
print("First 5 rows:") # prints the first 5 rows
print(data.head())

# Task 3: Display last 5 rows of the dataset
print("\nLast 5 rows:") # prints the last 5 rows
print(data.tail())

# Task 4: Determine shape of the dataset (shape - total numbers of rows and columns)
print("\nShape of dataset (rows, columns):") # prints the shape of the dataset
print(data.shape)

# Task 5: Display concise summary of the dataset (info)
print("\nDataset info:") # prints the info of the dataset
print(data.info())

# Task 6: Check the null values in dataset (isnull)
print("\nNull values in each column:") # prints the null values in each column
print(data.isnull().sum())

# Task 7: Identify library to plot graph to understand relations among various columns
# creates a grid of small graphs ea of the graphs is a grid comparing 2 columns
sns.pairplot(data) # creates a grid of small graphs
plt.show()

# Task 8: Create input dataset from original dataset by dropping irrelevant features
# Task 9: Create output dataset from original dataset
def split_data(data): # function called split_data
    # creates input data
    dropped_columns = ['Client Name', 'Client e-mail', 'Net Worth'] # drops only personal identifiers and target
    X = data.drop(dropped_columns, axis=1)
    # creates output data
    target_column = 'Net Worth' # target column
    Y = data[target_column] # output data
    return X, Y # returns the input and output data

def preprocess_data(data, test_size=0.2, random_state=42): # function called preprocess_data
    """
    Preprocess data: split, scale, and create train/test sets
    
    Args:
        data: Raw dataset
        test_size: Proportion of data for testing (default 0.2)
        random_state: Random seed for reproducibility (default 42)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler_X, scaler_y, X, Y, label_encoders)
    """
    # Split data into input and output
    X, Y = split_data(data)
    
    # Encode categorical variables
    categorical_columns = ['Profession', 'Education', 'Country', 'Gender']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Task 10: Transform input dataset into percentage based weighted between 0 and 1
    scaler_X = MinMaxScaler() # creates a scaler
    X_scaled = scaler_X.fit_transform(X) # fits the scaler to the input data
    
    # Task 11: Transform output dataset into percentage based weighted between 0 and 1
    scaler_y = MinMaxScaler() # creates a scaler
    y_scaled = scaler_y.fit_transform(Y.values.reshape(-1, 1)) # fits the scaler to the output data
    
    # Task 14: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split( # splits the data into training and testing sets
        X_scaled, y_scaled, test_size=test_size, random_state=random_state # splits the data into training and testing sets
    )
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, X, Y, label_encoders # returns the training and testing sets

# Use the preprocessing function
X_train, X_test, y_train, y_test, scaler_X, scaler_y, X, Y, label_encoders = preprocess_data(data) # calls the function

# Task 12: Print first few rows of scaled input dataset
print("\nScaled input features:") # prints the scaled input features
print(pd.DataFrame(X_train, columns=X.columns).head()) # prints the first few rows of the scaled input features

# Task 13: Print first few rows of scaled output dataset
print("\nScaled output values:") # prints the scaled output values
print(pd.DataFrame(y_train, columns=['Car Purchase Amount']).head()) # prints the first few rows of the scaled output values

# Task 15: Print shape of test and training data
print("\nShapes of training and testing sets:") # prints the shape of the training and testing sets
print("X_train:", X_train.shape) # prints the shape of the training set
print("X_test :", X_test.shape) # prints the shape of the testing set
print("y_train:", y_train.shape) # prints the shape of the training set
print("y_test :", y_test.shape) # prints the shape of the testing set

# Task 16: Print first few rows of test and training data
print("\nFirst few rows of X_train:") # prints the first few rows of the training set
print(pd.DataFrame(X_train, columns=X.columns).head()) # prints the first few rows of the training set

print("\nFirst few rows of y_train:") # prints the first few rows of the training set
print(pd.DataFrame(y_train, columns=['Car Purchase Amount']).head()) # prints the first few rows of the training set
print("\n") # prints a new line

# Task 17: Import and initialize AI models
# Task 18: Create an instance of each model you have imported
# initialise models
lr = LinearRegression() # creates a linear regression model
ridge = Ridge() # creates a ridge regression model
lasso = Lasso() # creates a lasso regression model
dt = DecisionTreeRegressor() # creates a decision tree regression model
rf = RandomForestRegressor() # creates a random forest regression model
gbr = GradientBoostingRegressor() # creates a gradient boosting regression model
svr = SVR() # creates a support vector regression model
knn = KNeighborsRegressor() # creates a k-nearest neighbors regression model
etr = ExtraTreesRegressor() # creates an extra trees regression model
xgb = XGBRegressor() # creates an xgboost regression model

# Task 19: Train models using training data
# Task 20: Train the models using both training sets (input and output) with fit()
# Train models
lr.fit(X_train, y_train) # trains the linear regression model
ridge.fit(X_train, y_train) # trains the ridge regression model
lasso.fit(X_train, y_train) # trains the lasso regression model
dt.fit(X_train, y_train) # trains the decision tree regression model
rf.fit(X_train, y_train) # trains the random forest regression model
gbr.fit(X_train, y_train) # trains the gradient boosting regression model
svr.fit(X_train, y_train) # trains the support vector regression model
knn.fit(X_train, y_train) # trains the k-nearest neighbors regression model
etr.fit(X_train, y_train) # trains the extra trees regression model
xgb.fit(X_train, y_train) # trains the xgboost regression model

# Task 21: Prediction on test data
# Task 22: Use predict() on each model initialized against input test data only
# predict input data
pred_lr = lr.predict(X_test) # predicts the linear regression model
pred_ridge = ridge.predict(X_test) # predicts the ridge regression model
pred_lasso = lasso.predict(X_test) # predicts the lasso regression model
pred_dt = dt.predict(X_test) # predicts the decision tree regression model
pred_rf = rf.predict(X_test) # predicts the random forest regression model
pred_gbr = gbr.predict(X_test) # predicts the gradient boosting regression model
pred_svr = svr.predict(X_test) # predicts the support vector regression model
pred_knn = knn.predict(X_test) # predicts the k-nearest neighbors regression model
pred_etr = etr.predict(X_test) # predicts the extra trees regression model
pred_xgb = xgb.predict(X_test) # predicts the xgboost regression model

# Task 23: Evaluate model performance
# Task 24: mean_squared_error() to measure accuracy of prediction against test output data
# Task 25: Needs to be repeated for each model trained
# compare actual vs predicted data`
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr)) # calculates the rmse for the linear regression model
rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge)) # calculates the rmse for the ridge regression model
rmse_lasso = np.sqrt(mean_squared_error(y_test, pred_lasso)) # calculates the rmse for the lasso regression model
rmse_dt = np.sqrt(mean_squared_error(y_test, pred_dt)) # calculates the rmse for the decision tree regression model
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf)) # calculates the rmse for the random forest regression model
rmse_gbr = np.sqrt(mean_squared_error(y_test, pred_gbr)) # calculates the rmse for the gradient boosting regression model
rmse_svr = np.sqrt(mean_squared_error(y_test, pred_svr)) # calculates the rmse for the support vector regression model
rmse_knn = np.sqrt(mean_squared_error(y_test, pred_knn)) # calculates the rmse for the k-nearest neighbors regression model
rmse_etr = np.sqrt(mean_squared_error(y_test, pred_etr)) # calculates the rmse for the extra trees regression model
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb)) # calculates the rmse for the xgboost regression model

# Task 26: Display evaluation results
# Task 27: Print the rmse values for each model
# Task 28: Rmse values provided from mean_squared_error result for each model
# display eval results
print("\nEvaluation Results (RMSE):")
print(f"Linear Regression: {rmse_lr:.4f}")
print(f"Ridge: {rmse_ridge:.4f}")
print(f"Lasso: {rmse_lasso:.4f}")
print(f"Decision Tree: {rmse_dt:.4f}")
print(f"Random Forest: {rmse_rf:.4f}")
print(f"Gradient Boosting: {rmse_gbr:.4f}")
print(f"SVR: {rmse_svr:.4f}")
print(f"KNN: {rmse_knn:.4f}")
print(f"Extra Trees: {rmse_etr:.4f}")
print(f"XGBoost: {rmse_xgb:.4f}")

# Find the best model (lowest RMSE)
models = {
    'Linear Regression': (lr, rmse_lr),
    'Ridge': (ridge, rmse_ridge),
    'Lasso': (lasso, rmse_lasso),
    'Decision Tree': (dt, rmse_dt),
    'Random Forest': (rf, rmse_rf),
    'Gradient Boosting': (gbr, rmse_gbr),
    'SVR': (svr, rmse_svr),
    'KNN': (knn, rmse_knn),
    'Extra Trees': (etr, rmse_etr),
    'XGBoost': (xgb, rmse_xgb)
}

best_model_name = min(models.keys(), key=lambda x: models[x][1])
best_model, best_rmse = models[best_model_name]

print(f"\nBest Model: {best_model_name} with RMSE: {best_rmse:.4f}")

# Task 29: Choose best model
# retrain best model on all data no more 80/20
X_all = scaler_X.fit_transform(X)
y_all = scaler_y.fit_transform(Y.values.reshape(-1, 1)).ravel()

best_model.fit(X_all, y_all)

# Task 30: Visualize model results by creating a bar chart
# Task 31: Add RSME values on top of bars
# Task 32: Display chart
# visualise model result using bar chart
model_names = ['LR', 'Ridge', 'Lasso', 'DT', 'RF', 'GBR', 'SVR', 'KNN', 'ETR', 'XGB']
rmse_values = [rmse_lr, rmse_ridge, rmse_lasso, rmse_dt, rmse_rf, rmse_gbr, rmse_svr, rmse_knn, rmse_etr, rmse_xgb]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, rmse_values, color='skyblue')
plt.title('Model Comparison (RMSE)')
plt.ylabel('RMSE')
plt.xlabel('Model')

# Add RMSE labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Task 33: Save the model
# save the best model and scalers using dump
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Task 34: Load the model
# load best model and scaler using load
best_model = joblib.load('best_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Task 35: Gather user inputs
# gather user inputs
def get_user_input():
    all_entries = []
    while True:
        user_df = launch_multi_entry_gui()
        if user_df is None:
            print("User cancelled input.")
            break  # Exit loop but still return what we have
        else:
            print("User input received:")
            print(user_df)
            all_entries.append(user_df)

    # Always return data, even if cancelled
    if all_entries:
        combined_df = pd.concat(all_entries, ignore_index=True)
        print(f"\nReturning {len(combined_df)} rows from {len(all_entries)} sessions")
        return combined_df
    else:
        print("No user data collected.")
        return pd.DataFrame()  # Return empty DataFrame instead of None

# Task 36: Use model to make predictions based on user input
# Usage
all_data = get_user_input()
if not all_data.empty:
    print("Data available for use:")
    print(all_data)
    
    # Preprocess user data to match training data format
    all_data_processed = all_data.copy()
    
    # Encode categorical variables using the same encoders from training
    categorical_columns = ['Profession', 'Education', 'Country', 'Gender']
    for col in categorical_columns:
        if col in all_data_processed.columns and col in label_encoders:
            # Handle unseen categories by using a default value
            try:
                all_data_processed[col] = label_encoders[col].transform(all_data_processed[col])
            except ValueError:
                # If there are unseen categories, map them to a default value
                print(f"Warning: Found unseen categories in {col}, mapping to default value")
                all_data_processed[col] = 0  # Default to first category
    
    print("\nProcessed data (categorical variables encoded):")
    print(all_data_processed)
    
    # splitting input n output data for user input
    X_user, Y_user = split_data(all_data_processed)
    
    # Transform user input using the same scaler that was fitted on training data
    X_user_scaled = scaler_X.transform(X_user)  # Use transform, not fit_transform
    
    # use model to make predictions for the user input
    pred_lr_user = best_model.predict(X_user_scaled)
    
    # Inverse transform predictions to get actual values
    pred_lr_user_original = scaler_y.inverse_transform(pred_lr_user.reshape(-1, 1))
    
    # Display predictions
    print("\nPredictions for user input:")
    for i, (actual, predicted) in enumerate(zip(Y_user, pred_lr_user_original)):
        print(f"Entry {i+1}:")
        print(f"  Actual Net Worth: ${actual:.2f}")
        print(f"  Predicted Net Worth: ${predicted[0]:.2f}")
        print(f"  Difference: ${abs(actual - predicted[0]):.2f}")
        print()
    
    # Calculate RMSE for user predictions
    rmse_user = np.sqrt(mean_squared_error(Y_user, pred_lr_user_original))
    print(f"RMSE for user predictions: {rmse_user:.4f}")
    
else:
    print("No user data to make predictions on.")

# Task 37: Predict on new test data
# predict on new test data
pred_lr_new_test = best_model.predict(X_test)

# compare actual vs predicted data
rmse_new_test = np.sqrt(mean_squared_error(y_test, pred_lr_new_test))

# display eval results
print("\nEvaluation Results for new test data (RMSE):")
print(f"Linear Regression: {rmse_new_test:.4f}")