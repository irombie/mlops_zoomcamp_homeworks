import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import date
import datetime

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.orion.schemas.schedules import CronSchedule
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
import pickle

@task
def get_paths(date_input):
    if date_input is None:
        today = date.today()
        date_input = today.strftime("%Y-%m-%d")
    datem = datetime.datetime.strptime(date_input, "%Y-%m-%d")

    month = datem.month
    year = datem.year

    train_month = (datem.month - 2) % 12 
    val_month = (datem.month - 1) % 12 

    if(month<=2):
        train_year = year -1
    else:
        train_year = year

    if(month<=1):
        val_year = year -1
    else:
        val_year = year
    
    add_train = ""
    if(train_month<10):
        add_train += "0"
    

    add_val = ""
    if(val_month<10):
        add_val += "0"
    

    root_dir = './data/fhv_tripdata'

    train_date_enxtension = '{}-{}{}.parquet'.format(train_year, add_train, train_month)
    train_path = '{}_{}'.format(root_dir, train_date_enxtension) 
    val_date_enxtension = '{}-{}{}.parquet'.format(val_year, add_val,val_month)
    val_path = '{}_{}'.format(root_dir, val_date_enxtension) 

    return train_path, val_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task 
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df
@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv
@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    categorical = ['PUlocationID', 'DOlocationID']
    
    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    pickle.dump(lr, open("model-{}.bin".format(date), 'wb'))
    pickle.dump(dv, open("dv-{}.bin".format(date), 'wb'))
    run_model(df_val_processed, categorical, dv, lr)

main(date="2021-03-15")

# DeploymentSpec(
#     flow=main,
#     name="model_training",
#     schedule=CronSchedule(
#         cron=" 0 9 15 * *",
#         timezone="America/New_York"),
#         flow_runner=SubprocessFlowRunner(),
#         tags=["ml"]
# )