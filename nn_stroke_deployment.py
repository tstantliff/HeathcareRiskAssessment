import snowflake.connector
from snowflake.snowpark import Session
from snowflake.snowpark.types import FloatType
import pandas as pd
import os
from config import connection_parameters 

# Connection details
connection_parameters = {
    "account": "<YOUR_ACCOUNT>",
    "user": "<YOUR_USERNAME>",
    "password": "<YOUR_PASSWORD>",
    "warehouse": "<YOUR_WAREHOUSE>",
    "database": "<YOUR_DATABASE>",
    "schema": "<YOUR_SCHEMA>",
}

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import numpy as np
except ImportError as e:
    missing_package = str(e).split()[-1]
    raise ImportError(
        f"Missing package {missing_package}. Please install it using '!pip install {missing_package}'."
    )

# Establish Snowflake session
try:
    session = Session.builder.configs(connection_parameters).create()
    print("Successfully connected to Snowflake.")
except Exception as e:
    print(f"Error connecting to Snowflake: {e}")
    raise

try:
    model_path = os.path.join(project_root, 'model', 'nn_model.h5')
    session.file.put(model_path, "@ml_stage", auto_compress=False)
    print("Model uploaded successfully to Snowflake stage.")
except Exception as e:
    print(f"Error uploading the model to Snowflake stage: {e}")
    raise

# Define a function for loading the model and making predictions
def predict_stroke(session: Session, input_data: list) -> float:
    try:
        # Load the model from the stage
        model_file_path = '@ml_stage/nn_model.h5'
        model = load_model(model_file_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model from Snowflake stage: {e}")
        raise

    try:
        # Convert input data to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        # Make prediction
        prediction = model.predict(input_array)
        return float(prediction[0][0])
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

# Register the function as a Snowflake UDF
try:
    session.udf.register(
        func=predict_stroke,
        name="PREDICT_STROKE",
        input_types=[FloatType()] * X_train.shape[1],
        return_type=FloatType(),
        replace=True
    )
    print("UDF registered successfully in Snowflake.")
except Exception as e:
    print(f"Error registering UDF in Snowflake: {e}")
    raise

# Using the UDF.
try:
    query = """
    SELECT PREDICT_STROKE(ARRAY_CONSTRUCT(<feature_1>, <feature_2>, ..., <feature_15>)) AS stroke_prediction
    FROM stroke_dataset_1
    """
    results = session.sql(query).collect()
    print("Query executed successfully. Results retrieved.")
    print(results)
except Exception as e:
    print(f"Error executing query in Snowflake: {e}")
    raise
