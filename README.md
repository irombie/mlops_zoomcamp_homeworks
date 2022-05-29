# MlOps Zoomcamp Homeworks

I will be sharing the homeworks and possibly the project for the MLOps Zoomcamp I am participating at, that is hosted by DataTalks.Club. 

I will try to make the content as clear to understand as possible, please let me know if you have comments. If you notice an error, or if you have suggestions for improvement, send me a pull request! 

The link to the Zoomcamp is here: https://github.com/DataTalksClub/mlops-zoomcamp

## Homework 1 

Homework 1 is a notebook with detailed instructions. Therefore, I will not include much details about it here. 

## Homework 2 

For homework 2, we used [MLflow](https://www.mlflow.org) for experiment tracking. 

This code uses the NYC Green Taxi dataset, which can be downloaded [here](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Then download Jan, Feb, March records of 2021.

The file ```preprocess_data.py``` is used to preprocess the data so it is in an appropriate shape for training. You need to run this file first before moving on to the MLflow parts. For running it, you need to specify a flag for the raw data path and the destination data path, which will be the path of the processed and training-ready data. 

For all of the remaining files, we need to specify the destination data path so the processed data can be accessed. 


We then train the model while also monitoring it using autolog feature of MLflow. We use ```train.py```. 

Then, we integrate [hyperopt](http://hyperopt.github.io/hyperopt/), which is a hyper-parameter optimization library. We search the parameter space easily using it and use MLflow to save and then cherry pick the models with good results, so they will be used in production later on. We use ```hpo.py```.

Finally, we pick the best model that is suited for our needs. In ```register_model.py``` file, we use MLflow to pick the model with the lowest Root Mean Squared Error (or RMSE for short) and  save it to the model registry. Don't know what model registry is? No problem, let us briefly explain it first:

A model registry is a model versioning system that keeps track of the good models that we save with their appropriate versions. Then, the appropriate version will be moved along for production. If something goes with the newly deployed model, we can easily locate the previous version of the model in the model registry and revert back to it with ease.

One more note before I stop blabbing about this: Just because a model has the lowest error value does not automatically make it the most suitable model. There might be resource constraints such as memory or time, that are also of utmost importance. Luckily, Mlflow also gives us a chance to monitor the amount of time it takes to train the model and the size of the model. We can then pick the best model for our needs! Way cool! 

To get started, all you need to do is setup MLflow using pip, conda or whichever package manager you prefer, run command ```mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root artifacts --host 0.0.0.0:80``` and surround your training codes with ```with mlflow.start_run():``` so MLflow can get to trackin'!
