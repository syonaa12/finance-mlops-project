from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='finance_train_pipeline',
    start_date=datetime(2024,1,1),
    schedule_interval='@daily',
    catchup=False
):
    train = BashOperator(
        task_id='run_training',
        bash_command='python /mnt/data/finance_mlops_repo/src/train.py'
    )
