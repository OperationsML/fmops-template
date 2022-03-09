##################################################  START BOILERPLATE CODE ##################################################
from flyermlops.pipelines.flight_utils import load_flight_metadata
from flyermlops.tracking.tracking_objects import DataTracker

import os

print(os.environ)

# Load config file (this is from the root directory - it will be included in every processing/training job)
config = load_flight_metadata()
print("loaded config")
# data tracking object is passed config (tracker uses aurora dev uri)
data_tracker = DataTracker(log=True, **config)

# Similar to mlflow, we create a data tracking id
data_tracker.start_tracking()

################################################## END BOILERPLATE CODE ##################################################

import pandas as pd
import awswrangler as wr
import numpy as np
import joblib

# Create function that can be called under the main clause. Or write everything under __main__
def load_data():

    print(f"Data tracking id for load-model: {data_tracker.tracking_id}")
    print(f"Flight tracking: {data_tracker.flight_tracking_id}")

    records = 20000
    cols = 10
    mu_1 = -4  # mean of the first distribution
    mu_2 = 4  # mean of the second distribution
    X_train = np.round(np.random.normal(mu_1, 2.0, size=(records, cols)), 4)
    X_test = np.round(np.random.normal(mu_1, 2.0, size=(records, cols)), 4)

    col_names = []
    for i in range(0, X_train.shape[1]):
        col_names.append(f"col_{i}")

    X_train = pd.DataFrame(X_train, columns=col_names)
    X_train["target_feature"] = np.random.randint(1, 100, size=(records, 1))

    X_test = pd.DataFrame(X_test, columns=col_names)

    # Add noise to feature to detect in data drift
    X_test["col_0"] = np.random.rand(X_test.shape[0]) * X_test["col_0"]
    X_test["target_feature"] = np.random.randint(1, 100, size=(records, 1))

    # Run drift diagnostics
    # This will create drift measurements on your two dataset that can be viewed within mlflow when the new model is saved.
    data_tracker.run_drift_diagnostics(
        reference_data=X_train, current_data=X_test, target_feature="target_feature",
    )

    # Simialr to mlflow you can log any object as an artifact. You will need to save it locally first and then log it after.
    for data, name in zip([X_train, X_test], ["train.joblib", "test.joblib"]):
        joblib.dump(data, name)
        data_tracker.log_artifact(name, version=True)

    # End tracking
    data_tracker.end_tracking()


if __name__ == "__main__":

    try:
        # Run function
        load_data()

    except Exception as e:
        # End tracking
        data_tracker.end_tracking()
        raise e
