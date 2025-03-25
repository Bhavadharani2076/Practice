from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_stress_detector(df):
    """Train a stress detection model."""
    # Features and labels
    X = df[["Temperature (°C)", "Systolic_BP (mmHg)", "Diastolic_BP (mmHg)", "Heart_Rate (bpm)"]]
    y = df["Target_Health_Status"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model to a file
    joblib.dump(clf, "stress_detection_model.pkl")
    print("Model saved to stress_detection_model.pkl")

    return clf

def detect_stress(clf, data):
    """Detect stress using the trained model."""
    return clf.predict([data])[0]