version = "1.0"

train {
    step generate_features {
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [{sh = ["python3 generate_features.py"]}]
        resources {
            cpu = "2"
            memory = "10G"
        }
    }

    step train {
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "2"
            memory = "10G"
        }
        depends_on = ["generate_features"]
    }

    parameters {
        FEATURES_FILE = "FEATURES_FILE_GZ"  // in this example, ensure a unique name to avoid name clashes with other users
        N_ESTIMATORS = "400"
        MAX_FEATURES = "sqrt"
        BOOTSTRAP = "FALSE"
        RANDOM_STATE = "42"
    }
}