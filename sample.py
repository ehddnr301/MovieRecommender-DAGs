from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash_operator import BashOperator

# 기본 DAG 설정
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG 인스턴스 생성
dag = DAG(
    "sample_dag",
    default_args=default_args,
    description="A simple tutorial DAG",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# 더미 작업 생성
start = DummyOperator(
    task_id="start",
    dag=dag,
)

end = DummyOperator(
    task_id="end",
    dag=dag,
)

# Bash 작업 생성
task1 = BashOperator(
    task_id="run_task1",
    bash_command='echo "Hello from Task 1"',
    dag=dag,
)

task2 = BashOperator(
    task_id="run_task2",
    bash_command='echo "Hello from Task 2"',
    dag=dag,
)

task3 = BashOperator(
    task_id="run_task3",
    bash_command='echo "Hello from Task 3"',
    dag=dag,
)

# 의존성 설정
start >> task1 >> [task2, task3] >> end
