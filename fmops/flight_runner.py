from flyermlops.pipelines import run_flight
import argparse


def load_data(start_flight):
    entry_point = "load_data.py"
    ac_type = "sklearn"
    engine = "ml.m5.large"
    number_engines = 1
    retry = 0


def lasso_model(load_data):
    entry_point = "lasso_model.py"
    ac_type = "sklearn"
    engine = "ml.m5.large"
    number_engines = 1
    retry = 0


def lightgbm_model(load_data):
    entry_point = "lightgbm_model.py"
    image_uri = "080196648158.dkr.ecr.us-east-1.amazonaws.com/ods-mlops-lightgbm:latest"
    ac_type = "sklearn"
    engine = "ml.m5.large"
    number_engines = 1
    retry = 0


flight_list = [load_data, lasso_model, lightgbm_model]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Args used for cicd
    parser.add_argument("--show_diagram", default=False)
    parser.add_argument("--save_diagram", default=False)
    parser.add_argument("--run_flight", default=True)
    parser.add_argument("--sm_role", default=None)

    args, _ = parser.parse_known_args()

    run_flight(
        sm_role=args.sm_role,
        flight_route=flight_list,
        show_diagram=args.show_diagram,
        save_diagram=args.save_diagram,
        run_flight=True,
    )
