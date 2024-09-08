import argparse
import pandas as pd
import keras
import joblib
from sklearn.model_selection import train_test_split

# Load the models
dl_model = keras.models.load_model('DL_Ver1.ipynb')
lr_model = joblib.load('logistic_regression_model.pkl')
lstm_model = keras.models.load_model('lstm_model.keras')
rf_model = joblib.load('random_forest_model.pkl')

# Define a function to make predictions using each model
def predict_failure_date(serial_number):
    # Load the preprocessed data
    data = pd.read_csv('preprocessed_data.csv')

    # Filter the data to select the row with the specified serial number
    disk_data = data[data['serial_number'] == serial_number]

    # Extract the features from the data and reshape them as needed
    features = disk_data[['capacity_bytes', 'smart_5_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw',
                          'smart_188_raw', 'smart_190_raw', 'smart_193_raw', 'smart_194_raw',
                          'smart_197_raw', 'smart_198_raw', 'smart_199_raw', 'smart_241_raw', 'smart_242_raw']].values

    # Make predictions using each model
    dl_prediction = dl_model.predict(features)
    lr_prediction = lr_model.predict(features)
    lstm_prediction = lstm_model.predict(features.reshape(1, -1, features.shape[1]))
    rf_prediction = rf_model.predict(features)

    # Perform voting to determine the final prediction
    predictions = [dl_prediction, lr_prediction, lstm_prediction, rf_prediction]
    final_prediction = max(set(predictions), key=predictions.count)

    return final_prediction

# Define a function to perform additional training using each model
def train_models(csv_file):
    # Load the data from the CSV file
    data = pd.read_csv(csv_file)

    # Preprocess the data as needed

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data.drop('date_of_failure', axis=1), data['date_of_failure'], test_size=0.2, random_state=42)

    # Retrain each model on the new data
    dl_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    lr_model.fit(X_train, y_train)
    lstm_model.fit(X_train.values.reshape(-1, 1, X_train.shape[1]), y_train, validation_data=(X_val.values.reshape(-1, 1, X_val.shape[1]), y_val), epochs=10, batch_size=32)
    rf_model.fit(X_train, y_train)

    # Save the retrained models to files
    dl_model.save('DL_Ver1.ipynb')
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    lstm_model.save('lstm_model.keras')
    joblib.dump(rf_model, 'random_forest_model.pkl')

# Define the CLI interface using argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", help="input file")
parser.add_argument("-dI", "--diskInfo", action="store_true", help = "Enter the serial number of the disk, which you want to know when this disk fails")
parser.add_argument("-l", "--learn", action="store_true", help="Enter the path to the csv file to further train the model")
args = parser.parse_args()

# Handle the -dI and -l options
if args.diskInfo:
    failure_date = predict_failure_date(args.input)
    print(f"Your disk {args.input} fails at {failure_date}")
elif args.learn:
    train_models(args.input)
    print(f"AI started additional training from file {args.input}")
else:
    print("Wrong action!")
