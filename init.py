import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from minio import Minio
from minio.error import S3Error
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split


def init_data():
    engine = create_engine(
        "postgresql://postgres:postgres@postgres-service:5432/postgres", echo=False
    )

    movies_df = pd.read_csv("/opt/airflow/dags/etl-dags/data/init_movie.csv")
    movies_df = movies_df.drop_duplicates()
    ratings_df = pd.read_csv("/opt/airflow/dags/etl-dags/data/init_rating.csv")

    movies_df["created_at"] = datetime.now()
    movies_df["updated_at"] = datetime.now()
    ratings_df["created_at"] = datetime.now()
    ratings_df["updated_at"] = datetime.now()

    movies_df.to_sql("movies", engine, if_exists="append", index=False)
    ratings_df.to_sql("ratings", engine, if_exists="append", index=False)


dag = DAG(
    "init_data",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="A simple DAG to upload CSV file to MinIO",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["insert", "initial", "data"],
    is_paused_upon_creation=False,
)

insert_init_data = PythonOperator(
    task_id="insert_init_data",
    python_callable=init_data,
    dag=dag,
)

insert_init_data
