version = "1.0"

train {
    step generate_features {
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "generate_features.py"
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.1"
                    spark.kubernetes.pyspark.pythonVersion = "3"
                    spark.driver.memory = "4g"
                    spark.driver.cores = "2"
                    spark.executor.instances = "2"
                    spark.executor.memory = "4g"
                    spark.executor.cores = "2"
                    spark.memory.fraction = "0.5"
                    spark.sql.parquet.compression.codec = "gzip"
                    spark.hadoop.fs.s3a.impl = "org.apache.hadoop.fs.s3a.S3AFileSystem"
                    spark.hadoop.fs.s3a.endpoint = "s3.ap-southeast-1.amazonaws.com"
                }
                settings {
                }
            }}
        ]
        resources {
            cpu = "0.5"
            memory = "1G"
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
        RANDOM_STATE = "42
    }
}