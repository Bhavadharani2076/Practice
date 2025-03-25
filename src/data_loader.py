import pandas as pd

def load_dataset(file_path):
    """Load the dataset from the CSV file."""
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(df)} rows.")
    return df

def preprocess_data(df):
    """Preprocess the dataset."""
    # Drop unnecessary columns
    df = df.drop(columns=["Patient_ID", "Timestamp", "Sensor_ID", "Sensor_Type", "Device_Battery_Level (%)"])

    # Convert Target_Health_Status to binary labels (1 = Unhealthy, 0 = Healthy)
    df["Target_Health_Status"] = df["Target_Health_Status"].apply(lambda x: 1 if x == "Unhealthy" else 0)

    return df