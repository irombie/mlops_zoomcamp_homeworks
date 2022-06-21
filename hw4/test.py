import requests
import json
import pandas as pd
from tqdm import tqdm

categorical = ['PULocationID', 'DOLocationID']
numerical = ["trip_distance"]
def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/green_tripdata_2021-04.parquet')
# print(df.head())

rides = df[categorical+numerical].to_dict(orient='records')


url='http://127.0.0.1:9696/predict'
sum = 0
count = 0
for ride in tqdm(rides):
    # print(ride)
    # raise Exception("dur")
    response = requests.post(url, json=ride)
    sum += response.json()['duration']
    count += 1


print(sum/count)