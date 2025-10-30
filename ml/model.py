import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier

# Define the train_model function
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Implement the function
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
    
# Define the compute_model_metrics function
def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

# Define the inference function
def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Implement the function
    preds = model.predict(X)  # Use the model to make predictions on test data
    return preds
    
# Define the save_model function
def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # Implement the function
    with open(path, 'wb') as f:
        pickle.dump(model, f)

# Define the load_model function
def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # Implement the function
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data    

# Define the performance_on_categorical_slice function
def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : RandomForestClassifier
        Trained machine learning
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    
    # Create a data frame slice with only the specified slice_value data
    data_slice = data[data[column_name] == slice_value]
    # Separate the label and feature data in data slice
    # Process data slice
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features,
        label,
        training=False,
        encoder=encoder,
        lb=lb
    )
    # Get prediction on x_slice using inference function
    preds = model.predict(X_slice)  # Get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
