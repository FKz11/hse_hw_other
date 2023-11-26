"""
В рамках подготовки домашнего задания все имена переменных могут быть любыми (когда тестируете у себя). 
Но при сдаче домашнего задания на проверку просьба не менять следующие пункты:
  - BUCKET оставить как есть в этом шаблоне;
  - EXPERIMENT_NAME и DAG_ID оставить как есть (ссылками на переменную NAME);
  - имена коннекторов: pg_connection и s3_connection;
  - данные должны читаться из таблицы с названием california_housing;
  - данные на S3 должны лежать в папках {NAME}/datasets/ и {NAME}/results/.
"""
import json
import logging
import mlflow
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Literal

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

NAME = "FK_z11"
BUCKET = "lizvladi-mlops"
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"
EXPERIMENT_NAME = NAME
DAG_ID = NAME

models = dict(zip(["rf", "lr", "hgb"], [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))
default_args = {
    "owner": "Konstantin Sergeev",
    "email": "735rjcnz735@gmail.com",
    "email_on_failure": True,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}
dag = DAG(dag_id=DAG_ID,
          default_args=default_args,
          schedule_interval="0 1 * * *",
          start_date=days_ago(2),
          catchup=False,
          tags=["mlops", NAME]
          )


def init() -> Dict[str, Any]:
    metrics = {}
    metrics["start_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    experiment_name = EXPERIMENT_NAME
    experiments_already = [experiment.name for experiment in mlflow.search_experiments()]
    num_copy = 1
    while experiment_name in experiments_already:
        experiment_name = EXPERIMENT_NAME + f' ({num_copy})'
        num_copy += 1
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=f"s3://kosserg-mlflow-artifacts/{experiment_name}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name = 'PARENT_RUN', experiment_id = experiment_id, description = "PARENT_RUN") as parent_run:
        run_id = parent_run._info.run_id

    metrics["run_id"] = run_id
    metrics["experiment_name"] = experiment_name
    metrics["experiment_id"] = experiment_id

    return metrics


def get_data_from_postgres(**kwargs) -> Dict[str, Any]:

    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")
    metrics["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()

    data = pd.read_sql_query("SELECT * FROM california_housing", con)

    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    file_name = f"{NAME}/datasets/california_housing.pkl"
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, file_name).put(Body=pickle_byte_obj)

    _LOG.info("Data download finished.")
    
    metrics["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:

    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="get_data")
    metrics["data_preparation_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    s3_hook = S3Hook("s3_connection")
    
    file_name = f"{NAME}/datasets/california_housing.pkl"
    file = s3_hook.download_file(key=file_name, bucket_name=BUCKET)
    data = pd.read_pickle(file)

    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(["X_train", "X_test", "y_train", "y_test"],
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET,
                        f"{NAME}/datasets/{name}.pkl").put(Body=pickle_byte_obj)

    _LOG.info("Data preparation finished.")
    
    metrics["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics

def train_mlflow_model(model: Any, name: str, X_train: np.array,
                       X_test: np.array, y_train: pd.Series,
                       y_test: pd.Series) -> None:

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    signature = mlflow.models.infer_signature(X_test, prediction)

    model_info = mlflow.sklearn.log_model(model, name, signature=signature)

    mlflow.evaluate(
        model_info.model_uri,
        data=X_test,
        targets=y_test.values,
        model_type="regressor",
        evaluators=["default"],
    )

    
def train_model(**kwargs) -> Dict[str, Any]:

    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="prepare_data")
    model_name = kwargs["model_name"]
    run_id = metrics["run_id"]
    experiment_id = metrics["experiment_id"]

    s3_hook = S3Hook("s3_connection")

    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"{NAME}/datasets/{name}.pkl",
                                     bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    with mlflow.start_run(run_id=run_id) as parent_run:
        with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, nested=True) as child_run:
            metrics[f"train_start_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")
            train_mlflow_model(models[model_name], model_name, data["X_train"], data["X_test"], data["y_train"], data["y_test"])
            metrics[f"train_end_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics


def save_results(**kwargs) -> None:

    ti = kwargs["ti"]
    models_metrics = ti.xcom_pull(task_ids=[f"train_{model_name}" for model_name in models.keys()])
    result = {}
    for model_metrics in models_metrics:
        result.update(model_metrics)

    result["end_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    date = datetime.now().strftime("%Y_%m_%d_%H")
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    json_byte_object = json.dumps(result)
    file_name = f"{NAME}/results/{date}.json"
    resource.Object(
        BUCKET,
        file_name).put(Body=json_byte_object)
    


#################################### INIT DAG ####################################

task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id="get_data",
                               python_callable=get_data_from_postgres,
                               dag=dag,
                               provide_context=True)

task_prepare_data = PythonOperator(task_id="prepare_data",
                                   python_callable=prepare_data,
                                   dag=dag,
                                   provide_context=True)

task_train_models = [
    PythonOperator(task_id=f"train_{model_name}",
                   python_callable=train_model,
                   dag=dag,
                   provide_context=True,
                   op_kwargs={"model_name": model_name})
    for model_name in models.keys()
]

task_save_results = PythonOperator(task_id="save_results",
                                   python_callable=save_results,
                                   dag=dag,
                                   provide_context=True)

task_init >> task_get_data >> task_prepare_data >> task_train_models >> task_save_results