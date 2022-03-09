##################################################  START BOILERPLATE CODE ##################################################
from flyermlops.pipelines.flight_utils import get_config
from flyermlops.tracking.tracking_objects import ModelTracker

config = get_config()

# Similar to data tracker, model tracker is used to track object (models in this case)
# Importantly, ModelTracker is wrapper for MLFlow. It adds additional functionality as well keeps data and models tied to the same pipeline.
# This makes it easier to pull and log data and model artifacts
model_tracker = ModelTracker(log=True, **config)
model_tracker.start_tracking()
################################################## END BOILERPLATE CODE ##################################################


from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# Define train function to be run under __main__ clause. You can also write all your code under __main__ as well.
def train_model():
    print(f"Model tracking id: {model_tracker.tracking_id}")
    print(f"Model data tracking id for load-model: {model_tracker.data_tracking_id}")
    print(f"Flight tracking id: {model_tracker.flight_tracking_id}")

    # Load data
    model_tracker.load_artifact("train.joblib", "artifacts/train.joblib")
    train_df = joblib.load("artifacts/train.joblib")

    # Load data
    model_tracker.load_artifact("test.joblib", "artifacts/test.joblib")
    test_df = joblib.load("artifacts/test.joblib")

    X_train, y_train = train_df, np.array(train_df.pop("target_feature"))
    X_test, y_test = test_df, np.array(test_df.pop("target_feature"))

    # Standard scaler
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    reg = LGBMRegressor()
    reg.fit(X_train_std, y_train.ravel())

    y_pred = reg.predict(X_test_std)

    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    for key, val in zip(["rmse", "r2", "mae"], [rmse, r2, mae]):
        model_tracker.log_metric(key, val)

    # Save preprocessor

    # Save preprocessor.
    # It is recommended to save everything in the artifacts directory at the moment
    preprocessor_filename = "artifacts/preprocessor.joblib"
    joblib.dump(scaler, preprocessor_filename)
    model_tracker.log_artifact(preprocessor_filename, "artifacts")

    signature = model_tracker.infer_schema(X_train_std, reg.predict(X_test_std))

    # Save model
    # save model is a switch statment function that can log any model type that mlflow supports.
    # Refer to the function doc string for more information
    model_tracker.save_model(
        model_type="sklearn",
        model=reg,
        artifact_path="artifacts",
        registered_model_name="lightgbm",
        signature=signature,
        input_example=test_df[:100],
    )

    model_tracker.end_tracking()


if __name__ == "__main__":

    try:
        train_model()
    except Exception as e:
        model_tracker.end_tracking()
        raise e
