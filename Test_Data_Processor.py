import pandas as pd
import pytest
import numpy as np
from Regression_Models import split_data, load_data, preprocess_data
from sklearn.preprocessing import MinMaxScaler

# Load data once for all tests
@pytest.fixture
def data():
    return load_data()

@pytest.fixture
def preprocessed_data(data):
    """Fixture that provides preprocessed data for testing"""
    return preprocess_data(data)

def test_input_data_does_not_contain_sensitive_columns(data):
    """
    Test Case 1: Input data - Verifying the input data doesn't contain unnecessary columns
    Tests that sensitive columns (Client Name, Client e-mail) are dropped from input data
    """
    X, _ = split_data(data)
    
    # Check that sensitive columns are NOT in input data
    assert "Client Name" not in X.columns, "Client Name should be dropped from input data"
    assert "Client e-mail" not in X.columns, "Client e-mail should be dropped from input data"
    
    # Check that relevant features remain (all except personal identifiers and target)
    expected_features = ['Profession', 'Education', 'Country', 'Gender', 'Age', 'Income', 
                        'Credit Card Debt', 'Healthcare Cost', 'Inherited Amount', 
                        'Stocks', 'Bonds', 'Mutual Funds', 'ETFs', 'REITs']
    for feature in expected_features:
        assert feature in X.columns, f"Feature {feature} should be in input data"

def test_output_data_contains_only_target_column(data):
    """
    Test Case 2: Output data - Verifying the output data contains only the target column
    Tests that output data only contains the Net Worth column
    """
    _, Y = split_data(data)
    
    # Check that output is a Series (single column) with correct name
    assert isinstance(Y, pd.Series), "Output should be a pandas Series"
    assert Y.name == 'Net Worth', "Output should be named 'Net Worth'"
    
    # Check that no other columns are present in output
    assert len(Y.shape) == 1, "Output should be 1-dimensional"
    
    # Verify no sensitive or feature columns are in output
    sensitive_columns = ["Client Name", "Client e-mail"]
    feature_columns = ['Profession', 'Education', 'Country', 'Gender', 'Age', 'Income', 
                      'Credit Card Debt', 'Healthcare Cost', 'Inherited Amount', 
                      'Stocks', 'Bonds', 'Mutual Funds', 'ETFs', 'REITs']
    
    # Since Y is a Series, we check it doesn't contain any of these as column names
    for col in sensitive_columns + feature_columns:
        assert col not in [Y.name], f"Column {col} should not be in output data"

def test_shape_of_data_is_correct(data):
    """
    Test Case 3: Shape of data - Verifying that the shape of the data is correct 
    and dropping columns hasn't been performed incorrectly
    """
    X, Y = split_data(data)
    
    # Check input data shape (14 features: all except personal identifiers and target)
    expected_input_columns = 14  # Profession, Education, Country, Gender, Age, Income, 
                                 # Credit Card Debt, Healthcare Cost, Inherited Amount,
                                 # Stocks, Bonds, Mutual Funds, ETFs, REITs
    assert X.shape[1] == expected_input_columns, f"Input should have {expected_input_columns} columns, got {X.shape[1]}"
    assert X.shape[0] == data.shape[0], "Input should have same number of rows as original data"
    
    # Check output data shape
    assert Y.shape[0] == data.shape[0], "Output should have same number of rows as original data"
    
    # Verify total columns dropped equals expected
    original_columns = len(data.columns)
    input_columns = X.shape[1]
    output_columns = 1  # Y is a Series, so 1 column
    
    # Calculate actual dropped columns
    actual_dropped_columns = original_columns - input_columns - output_columns
    
    # Expected dropped columns: Client Name, Client e-mail, Net Worth (but Net Worth becomes output)
    # So only 2 are actually "dropped" from input
    expected_dropped_columns = 2  # Client Name, Client e-mail
    
    assert actual_dropped_columns == expected_dropped_columns, f"Should have dropped {expected_dropped_columns} columns, but dropped {actual_dropped_columns}"
    
    # Verify the specific columns that should be dropped
    dropped_columns = ['Client Name', 'Client e-mail']
    for col in dropped_columns:
        assert col not in X.columns, f"Column '{col}' should be dropped from input data"
        assert col not in [Y.name], f"Column '{col}' should not be in output data"
    
    # Verify Net Worth is in output but not in input
    assert 'Net Worth' not in X.columns, "Net Worth should not be in input data"
    assert Y.name == 'Net Worth', "Net Worth should be the output target"

def test_preprocessing_function_returns_correct_shapes(preprocessed_data):
    """
    Test that the preprocessing function returns correctly shaped data
    """
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, X, Y, label_encoders = preprocessed_data
    
    # Check that all returned objects have correct types
    assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
    assert isinstance(X_test, np.ndarray), "X_test should be numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be numpy array"
    assert isinstance(y_test, np.ndarray), "y_test should be numpy array"
    assert isinstance(scaler_X, MinMaxScaler), "scaler_X should be MinMaxScaler"
    assert isinstance(scaler_y, MinMaxScaler), "scaler_y should be MinMaxScaler"
    assert isinstance(label_encoders, dict), "label_encoders should be a dictionary"
    
    # Check that training and test sets have correct proportions
    total_samples = X_train.shape[0] + X_test.shape[0]
    expected_train_size = int(total_samples * 0.8)  # 80% for training
    expected_test_size = int(total_samples * 0.2)   # 20% for testing
    
    assert X_train.shape[0] == expected_train_size, f"Training set should have {expected_train_size} samples"
    assert X_test.shape[0] == expected_test_size, f"Test set should have {expected_test_size} samples"
    
    # Check that features are scaled between 0 and 1
    assert X_train.min() >= 0, "Scaled features should be >= 0"
    assert X_train.max() <= 1, "Scaled features should be <= 1"
    assert y_train.min() >= 0, "Scaled target should be >= 0"
    assert y_train.max() <= 1, "Scaled target should be <= 1"

def test_preprocessing_function_handles_different_test_sizes(data):
    """
    Test that preprocessing function works with different test sizes
    """
    # Test with 30% test size
    X_train, X_test, y_train, y_test, _, _, _, _, _ = preprocess_data(data, test_size=0.3)
    
    total_samples = X_train.shape[0] + X_test.shape[0]
    expected_train_size = int(total_samples * 0.7)  # 70% for training
    expected_test_size = int(total_samples * 0.3)   # 30% for testing
    
    assert X_train.shape[0] == expected_train_size, f"Training set should have {expected_train_size} samples with 30% test size"
    assert X_test.shape[0] == expected_test_size, f"Test set should have {expected_test_size} samples with 30% test size"

def test_categorical_encoding_works_correctly(data):
    """
    Test that categorical variables are properly encoded
    """
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, X, Y, label_encoders = preprocess_data(data)
    
    # Check that label encoders exist for categorical columns
    categorical_columns = ['Profession', 'Education', 'Country', 'Gender']
    for col in categorical_columns:
        assert col in label_encoders, f"Label encoder for {col} should exist"
        assert hasattr(label_encoders[col], 'transform'), f"Label encoder for {col} should have transform method"
    
    # Check that categorical columns are encoded as integers
    for col in categorical_columns:
        if col in X.columns:
            # Check that the encoded values are integers
            assert X[col].dtype in ['int64', 'int32', 'int'], f"Column {col} should be encoded as integers"

def test_privacy_compliance(data):
    """
    Additional test: Verify privacy compliance by ensuring sensitive data is properly removed
    """
    X, Y = split_data(data)
    
    # Define sensitive information that should be protected under privacy act
    sensitive_info = ["Client Name", "Client e-mail"]
    
    # Check that none of the sensitive information appears in input data
    for sensitive_col in sensitive_info:
        assert sensitive_col not in X.columns, f"Sensitive column '{sensitive_col}' should be removed for privacy compliance"
    
    # Check that sensitive information doesn't appear in output data
    for sensitive_col in sensitive_info:
        assert sensitive_col not in [Y.name], f"Sensitive column '{sensitive_col}' should not be in output data"
    
    print("✅ Privacy compliance verified - sensitive columns have been properly removed")

def test_all_required_features_are_present(data):
    """
    Test that all required features for net worth prediction are present
    """
    X, Y = split_data(data)
    
    # All features that should be used for net worth prediction
    required_features = [
        'Profession', 'Education', 'Country', 'Gender', 'Age',  # Demographic features
        'Income', 'Credit Card Debt', 'Healthcare Cost', 'Inherited Amount',  # Financial features
        'Stocks', 'Bonds', 'Mutual Funds', 'ETFs', 'REITs'  # Investment features
    ]
    
    for feature in required_features:
        assert feature in X.columns, f"Required feature '{feature}' should be present in input data"
    
    print(f"✅ All {len(required_features)} required features are present for net worth prediction")

def test_target_variable_is_correct(data):
    """
    Test that the target variable is correctly set to Net Worth
    """
    _, Y = split_data(data)
    
    assert Y.name == 'Net Worth', "Target variable should be 'Net Worth'"
    assert isinstance(Y, pd.Series), "Target should be a pandas Series"
    assert len(Y) > 0, "Target should contain data"
    
    # Check that target values are numeric
    assert pd.api.types.is_numeric_dtype(Y), "Target variable should be numeric"
    
    print("✅ Target variable 'Net Worth' is correctly configured")
