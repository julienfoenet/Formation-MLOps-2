import os
import sys
from datetime import timedelta
import mlflow
from mlflow.client import MlflowClient

from airflow.decorators import dag, task
from sqlalchemy_utils.types.enriched_datetime.pendulum_date import pendulum

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # So that airflow can find config files


@dag(default_args={'owner': 'airflow'}, schedule=timedelta(weeks=4),
     start_date=pendulum.today('UTC').add(hours=-1))
def promote_model():
    @task
    def promote_model_task() -> str:
        client = MlflowClient(mlflow.get_tracking_uri())
        model_info = client.get_latest_versions('Default')[0]
        client.set_model_version_tag(
            name='tp6',
            version=model_info.version,
            key='version',
            value='latest'
        )
        
    promote_model_task()

promote_model_dag = promote_model()
