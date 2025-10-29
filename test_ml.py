import pytest
import os
import pandas as pd
from train_model import train, p
# TODO: V1 add necessary import

project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)


@pytest.fixture(scope="session")
def test_ml_data():
    full_data = data # Load original dataset
    train_data = train  # Load training dataset
    model_accuracy = p # Load model accuracy
    return full_data, train_data, model_accuracy

# TODO: V1 implement the second test. Change the function name and input as needed
def test_split_columns(test_ml_data):
    """
    # Tests that columns contained in train data match original data frame.
    """
    full_data, train_data, _ = test_ml_data

    assert set(full_data.columns) == set(train_data.columns), \
        f"Original data and train data columns are not consistent"
    
# TODO: V1 implement the second test. Change the function name and input as needed
def test_split_ratio(test_ml_data):
    """
    # Tests to determine if actual split ratio is the ratio expected from data split
    """
    full_data, train_data, _ = test_ml_data
    # Initialize expected ratio and tolerance
    expected_ratio = .20
    tolerance = .01
    # Initialaize original and split test data frame length, calculate difference and ratios
    data_len = full_data.shape[0] # Original data frame length
    train_len = train_data.shape[0] # Split data frame train set length
    split_difference = data_len - train_len # Calculate len difference between original data and train data
    actual_split_ratio = split_difference / data_len # Calculate ratio difference, should be 1.0 - expected_ratio
    
    # Assert that ratio difference should be within expected ratio +/- tolerance or provide fail message
    assert expected_ratio >= actual_split_ratio - tolerance and expected_ratio <= actual_split_ratio + tolerance, \
        f"Split ratio {actual_split_ratio:.2f} is not within the expected range of {expected_ratio - tolerance:.2f} to {expected_ratio + tolerance:.2f}"

# TODO: V1 implement the third test. Change the function name and input as needed
def test_model_accuracy(test_ml_data):
    """
    # Tests that model accuracy meets a minimum threshold
    """
    _, _, actual_model_accuracy = test_ml_data # Load actual model accuracy
    accuracy_threshold = .70 # Set model accuracy threshold
    
    # Assert that model fails if actual model accuracy is below set accuracy threshold
    assert actual_model_accuracy >= accuracy_threshold, \
        f"Actual model accuracy of {actual_model_accuracy:.2f} below threshold of {accuracy_threshold:.2f}."


   

