import os
import time
import random
from datetime import datetime, timedelta

import mlflow
import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

from minio import Minio
from minio.error import S3Error


# fmt: off
GROUP_B = [  1,   2,   3,   4,   5,   6,   7,  10,  15,  16,  17,  18,  19,
        20,  21,  23,  24,  25,  26,  27,  28,  29,  30,  33,  34,  35,
        36,  41,  42,  43,  46,  47,  48,  49,  50,  52,  57,  62,  63,
        64,  66,  67,  68,  72,  73,  74,  75,  76,  78,  79,  80,  81,
        82,  84,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98, 103,
       104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
       120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
       133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 154, 155,
       156, 166, 167, 168, 169, 170, 171, 172, 174, 175, 176, 177, 178,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 198, 199, 201, 202,
       203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
       216, 217, 219, 220, 221, 222, 223, 226, 227, 228, 229, 230, 231,
       232, 233, 234, 235, 236, 237, 245, 246, 249, 250, 251, 252, 253,
       254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267,
       268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
       281, 282, 283, 284, 288, 290, 291, 292, 293, 294, 295, 296, 297,
       298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
       311, 312, 313, 314, 315, 316, 317, 318, 322, 323, 324, 325, 326,
       327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 344,
       345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 357, 358, 359,
       360, 361, 362, 365, 366, 367, 368, 372, 373, 374, 375, 376, 377,
       378, 379, 380, 381, 382, 383, 384, 385, 387, 388, 389, 390, 391,
       392, 393, 395, 396, 397, 398, 399, 400, 401, 402, 405, 406, 407,
       408, 409]
# fmt: on


def create_new_mlflow_model(model_name: str) -> int:
    client = MlflowClient()

    filter_string = f"name='{model_name}'"
    results = client.search_model_versions(filter_string)

    if not results:
        client.create_registered_model(model_name)

        return 200

    return 409


def create_model_version(model_name: str, run_id: str, model_uri: str) -> str:
    client = MlflowClient()

    model_source = RunsArtifactRepository.get_underlying_uri(model_uri)
    model_version = client.create_model_version(model_name, model_source, run_id)

    return model_version.version


def update_registered_model(model_name: str, version: str) -> str:
    client = MlflowClient()
    production_model = None
    current_model = client.get_model_version(model_name, version)

    filter_string = f"name='{current_model.name}'"
    model_version_list = client.search_model_versions(filter_string)

    for mv in model_version_list:
        if mv.current_stage == "Production":
            production_model = mv

    if production_model is None:
        client.transition_model_version_stage(
            current_model.name, current_model.version, "Production"
        )
        production_model = current_model

        return "Production Model Registered"

    else:
        client.transition_model_version_stage(
            current_model.name,
            current_model.version,
            "Production",
            archive_existing_versions=True,
        )
        production_model = current_model

        return "Production Model Updated"


# MinIO Configuration
minio_client = Minio(
    "minio-service:9000",
    access_key=os.getenv("AWS_ACCESS_KEY_ID", "test_user_id"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test_user_password"),
    secure=False,  # Set True if MinIO server supports TLS
)


def upload_file_to_minio(bucket_name, file_path, object_name):
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        with open(file_path, "rb") as file_data:
            file_stat = os.stat(file_path)
            minio_client.put_object(
                bucket_name,
                object_name,
                file_data,
                file_stat.st_size,
                content_type="application/csv",
            )
        print(
            f"'{file_path}' is successfully uploaded as '{object_name}' to bucket '{bucket_name}'."
        )
    except S3Error as exc:
        print("Error occurred while uploading file to MinIO:", exc)


def download_file_from_minio(bucket_name, object_name, file_path):
    try:
        if minio_client.bucket_exists(bucket_name):
            response = minio_client.get_object(bucket_name, object_name)

            with open(file_path, "wb") as file_data:
                for data in response.stream(32 * 1024):
                    file_data.write(data)

            print(
                f"'{object_name}' is successfully downloaded as '{file_path}' from bucket '{bucket_name}'."
            )
        else:
            print(f"The bucket '{bucket_name}' does not exist.")

    except S3Error as exc:
        print("Error occurred while downloading file from MinIO:", exc)


# fmt: off
GROUP_B = [  1,   2,   3,   4,   5,   6,   7,  10,  15,  16,  17,  18,  19,
        20,  21,  23,  24,  25,  26,  27,  28,  29,  30,  33,  34,  35,
        36,  41,  42,  43,  46,  47,  48,  49,  50,  52,  57,  62,  63,
        64,  66,  67,  68,  72,  73,  74,  75,  76,  78,  79,  80,  81,
        82,  84,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98, 103,
       104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
       120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
       133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 154, 155,
       156, 166, 167, 168, 169, 170, 171, 172, 174, 175, 176, 177, 178,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 198, 199, 201, 202,
       203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
       216, 217, 219, 220, 221, 222, 223, 226, 227, 228, 229, 230, 231,
       232, 233, 234, 235, 236, 237, 245, 246, 249, 250, 251, 252, 253,
       254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267,
       268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
       281, 282, 283, 284, 288, 290, 291, 292, 293, 294, 295, 296, 297,
       298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
       311, 312, 313, 314, 315, 316, 317, 318, 322, 323, 324, 325, 326,
       327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 344,
       345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 357, 358, 359,
       360, 361, 362, 365, 366, 367, 368, 372, 373, 374, 375, 376, 377,
       378, 379, 380, 381, 382, 383, 384, 385, 387, 388, 389, 390, 391,
       392, 393, 395, 396, 397, 398, 399, 400, 401, 402, 405, 406, 407,
       408, 409]
# fmt: on


def extract_data():
    engine = create_engine(
        "postgresql://postgres:postgres@postgres-service-target:5432/postgres",
        echo=False,
    )

    movies_df = pd.read_sql("select movie_id, title, genres from movies", engine)
    ratings_df = pd.read_sql("select user_id, movie_id, rating from ratings", engine)

    movies_df.to_csv("/opt/airflow/dags/etl-dags/data/movies.csv", index=False)
    ratings_df.to_csv("/opt/airflow/dags/etl-dags/data/ratings.csv", index=False)

    upload_file_to_minio(
        "test",
        "/opt/airflow/dags/etl-dags/data/movies.csv",
        "movies.csv",
    )
    upload_file_to_minio(
        "test",
        "/opt/airflow/dags/etl-dags/data/ratings.csv",
        "ratings.csv",
    )


def transform_data():
    download_file_from_minio(
        "test",
        "movies.csv",
        "/opt/airflow/dags/etl-dags/data/movies_extract.csv",
    )
    download_file_from_minio(
        "test",
        "ratings.csv",
        "/opt/airflow/dags/etl-dags/data/ratings_extract.csv",
    )

    movies_df = pd.read_csv(
        "/opt/airflow/dags/etl-dags/data/movies_extract.csv"
    ).drop_duplicates()
    ratings_df = pd.read_csv(
        "/opt/airflow/dags/etl-dags/data/ratings_extract.csv"
    ).drop_duplicates()

    final_df = ratings_df.merge(movies_df, on="movie_id")
    genres = final_df["genres"].str.get_dummies(sep="|")

    final_df = pd.concat([final_df[["movie_id", "user_id", "rating"]], genres], axis=1)

    final_df = final_df[final_df["user_id"].isin(GROUP_B)]

    train_df, test_df = train_test_split(
        final_df, test_size=0.5, random_state=42, stratify=final_df["user_id"]
    )

    train_df.to_csv("/opt/airflow/dags/etl-dags/data/train.csv", index=False)
    test_df.to_csv("/opt/airflow/dags/etl-dags/data/test.csv", index=False)
    genres.to_csv("/opt/airflow/dags/etl-dags/data/geners.csv", index=False)

    upload_file_to_minio(
        "test",
        "/opt/airflow/dags/etl-dags/data/geners.csv",
        "geners.csv",
    )

    upload_file_to_minio(
        "test",
        "/opt/airflow/dags/etl-dags/data/train.csv",
        "train.csv",
    )

    upload_file_to_minio(
        "test",
        "/opt/airflow/dags/etl-dags/data/test.csv",
        "test.csv",
    )


def train_model():
    download_file_from_minio(
        "test",
        "geners.csv",
        "/opt/airflow/dags/etl-dags/data/geners.csv",
    )
    download_file_from_minio(
        "test",
        "train.csv",
        "/opt/airflow/dags/etl-dags/data/train_transform.csv",
    )
    download_file_from_minio(
        "test",
        "test.csv",
        "/opt/airflow/dags/etl-dags/data/test_transform.csv",
    )
    genres = pd.read_csv("/opt/airflow/dags/etl-dags/data/geners.csv")
    train_df = pd.read_csv("/opt/airflow/dags/etl-dags/data/train_transform.csv")
    test_df = pd.read_csv("/opt/airflow/dags/etl-dags/data/test_transform.csv")

    user_models = {}

    mlflow.set_experiment("rec_model")

    for user_id in train_df["user_id"].unique():
        user = train_df[train_df["user_id"] == user_id]
        X_train = user[genres.columns]
        y_train = user["rating"]

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        user_models[user_id] = reg

    for user_id, reg in user_models.items():
        if user_id in test_df["user_id"].unique():
            user = test_df[test_df["user_id"] == user_id].copy()
            X_test = user[genres.columns]
            y_test = user["rating"]
            y_pred = reg.predict(X_test)
            MODEL_NAME = f"rec_model_{user_id}"
            with mlflow.start_run(run_name=MODEL_NAME):
                mae = mean_absolute_error(y_test, y_pred)
                mlflow.log_metric("mae", mae)
                mlflow.log_param("genres", genres.columns.tolist())
                current_model = mlflow.sklearn.log_model(reg, MODEL_NAME)
                run_id, model_uri = current_model.run_id, current_model.model_uri

                create_new_mlflow_model(MODEL_NAME)
                model_version = create_model_version(MODEL_NAME, run_id, model_uri)
                update_registered_model(MODEL_NAME, model_version)


dag = DAG(
    "train_rec_model",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["minio", "upload", "csv", "train", "model"],
    is_paused_upon_creation=False,
)

read_task = PythonOperator(
    task_id="extract_data",
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id="transform_data",
    python_callable=transform_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

read_task >> transform_task >> train_task
