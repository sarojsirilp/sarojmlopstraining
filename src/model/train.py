# Import libraries
import argparse
import glob
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow


# Define main function
def main(args):
    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


# Function to load CSVs into DataFrame
def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")

    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")

    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# Function to split data
def split_data(df, test_size=0.30, random_state=0):
    X = df[
        [
            "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
            "TricepsThickness", "SerumInsulin", "BMI",
            "DiabetesPedigree", "Age"
        ]
    ].values

    y = df["Diabetic"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# Function to train model
def train_model(reg_rate, X_train, X_test, y_train, y_test):
    LogisticRegression(
        C=1 / reg_rate,
        solver="liblinear"
    ).fit(X_train, y_train)


# Parse CLI arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data",
        dest="training_data",
        type=str
    )
    parser.add_argument(
        "--reg_rate",
        dest="reg_rate",
        type=float,
        default=0.01
    )
    return parser.parse_args()


# Run the script
if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    args = parse_args()
    main(args)

    print("*" * 60)
    print("\n\n")
