"""
Script to preprocess raw data.
"""
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import featuretools as ft
import s3fs

RAW_BUCKET = "basisai-samples/sparkify"  # DO NOT MODIFY
TMP_BUCKET = "span-production-temp-data"  # DO NOT MODIFY

# We will save the output features data in S3
# Have a unique name to avoid name clashes with other users,
# since the file is overwritten each time it is saved
FEATURES_FILE = os.getenv("FEATURES_FILE_GZ")

AGG_PRIMITIVES = ["count", "num_unique", "mode", "mean", "sum", "max", "min"]


def preprocess():
    # import data
    df = pd.DataFrame()
    with open(f"s3a://{RAW_BUCKET}/tiny_sparkify_event_data.json") as f:
        transaction = [json.loads(line) for line in f]
        df = df.append(transaction, ignore_index = True)

    # remove rows with empty userid
    df = df.drop(df[df['userId']==''].index)
    
    # fill na
    df.fillna({'artist':'', 'song': '', 'length': 0.0}, inplace=True)
    
    # transform some of the features
    df['ts'] = df['ts'].apply(lambda x: datetime.fromtimestamp(x / 1e3))
    df['registration'] = df['registration'].apply(lambda x: datetime.fromtimestamp(x / 1e3))
    df['transactionDate'] = df['ts'].dt.date
    df['registrationDate'] = df['registration'].dt.date
    
    df['userAgent'] = df['userAgent'].apply(regroup_userAgent)
    
    df['transactionId'] = df.index
    df['sessionId'] = df['sessionId'].apply(lambda x: str(x))
    df['sessionKey'] = df['sessionId']+'-'+df['userId']

    # drop some of the columns
    df.drop(labels=['ts', 'registration', 'sessionId', 'lastName', 'firstName', 'location', 'auth', 'level', 'method', 'status'], axis=1, inplace=True)

    # set anchor period, prediction window, offset
    min_date = min(df["transactionDate"])
    max_date = max(df["transactionDate"])
    anchor_period = 14
    prediction_window = 7
    offset = 3

    # generate features using dfs
    df_full = pd.DataFrame()
    start_date = min_date
    unique_page_lst = df["page"].unique().tolist()
    while start_date < max_date:
        # obtain data
        end_date = start_date + timedelta(anchor_period-1)
        prediction_start_date = end_date + timedelta(1)
        prediction_end_date = prediction_start_date + timedelta(prediction_window-1)

        if prediction_end_date > max_date:
            break
        else:
            is_within_current_period = (df['transactionDate']>=start_date) & (df['transactionDate'] <= end_date)
            is_within_prediction_period = (df['transactionDate']>=prediction_start_date) & (df['transactionDate'] <= prediction_end_date)
            df_subset = df.loc[is_within_current_period, :]
            prediction_page = df.loc[is_within_prediction_period, ['userId', 'page']]
            prediction_page['is_cancel'] = prediction_page['page'].apply(lambda x: 1 if x=='Cancellation Confirmation' else 0)
            labels = prediction_page.groupby(by=['userId'])[['is_cancel']].sum()
            labels_dict = labels.to_dict()
            active_users = list(labels_dict['is_cancel'].keys())
            # obtain features
            df_temp = dfs_features(df_subset, AGG_PRIMITIVES, unique_page_lst)
            df_temp['is_active'] = df_temp['userId'].apply(lambda x: x in active_users)
            # obtain labels
            df_temp['is_cancel'] = [0]*len(df_temp)
            df_temp['is_cancel'].loc[df_temp['is_active']==True] = df_temp['userId'].loc[df_temp['is_active']==True].apply(lambda x: 1 if labels_dict['is_cancel'][x] else 0)
            # drop userid
            df_temp.drop(labels=['userId', 'is_active'], axis=1, inplace=True)
            # append values
            df_full = df_full.append(df_temp)
            # offset
            start_date = start_date + timedelta(offset)
    
    # reset index
    df_full.reset_index(inplace=True)
    df_full.drop(labels = ['index'], axis = 1, inplace=True)

    # drop columns with 0 variance
    for col in df_full.columns:
        if df_full[col].nunique()==1:
            df_full.drop(labels=[col], axis=1, inplace=True)
       
    # drop irrelevant columns
    df_full.drop(labels=[
    'MODE(transaction.artist)',
    'MODE(transaction.song)',
    'MEAN(transaction.itemInSession)',
    'SUM(transaction.itemInSession)',
    'MIN(transaction.itemInSession)',
    'DAY(first_session_time)',
    'DAY(registrationDate)',
    'YEAR(registrationDate)',
    'MONTH(first_session_time)',
    'MONTH(registrationDate)',
    'NUM_UNIQUE(session.DAY(first_transaction_time))',
    'NUM_UNIQUE(session.MODE(transaction.artist))',
    'NUM_UNIQUE(session.MODE(transaction.page))',
    'NUM_UNIQUE(session.MODE(transaction.song))',
    'NUM_UNIQUE(session.MONTH(first_transaction_time))',
    'NUM_UNIQUE(session.WEEKDAY(first_transaction_time))',
    'MODE(session.DAY(first_transaction_time))',
    'MODE(session.MODE(transaction.artist))',
    'MODE(session.MODE(transaction.song))',
    'MODE(session.MONTH(first_transaction_time))',
    'MODE(session.WEEKDAY(first_transaction_time))',
    'MEAN(session.MEAN(transaction.itemInSession))',
    'MEAN(session.MIN(transaction.itemInSession))',
    'MEAN(session.SUM(transaction.itemInSession))',
    'SUM(session.MEAN(transaction.itemInSession))',
    'SUM(session.MIN(transaction.itemInSession))',
    'SUM(session.NUM_UNIQUE(transaction.page))',
    'SUM(session.NUM_UNIQUE(transaction.song))',
    'MAX(session.MEAN(transaction.itemInSession))',
    'MAX(session.MEAN(transaction.length))',
    'MAX(session.MIN(transaction.itemInSession))',
    'MAX(session.MIN(transaction.length))',
    'MAX(session.SUM(transaction.itemInSession))',
    'MIN(session.MEAN(transaction.itemInSession))',
    'MIN(session.SUM(transaction.itemInSession))',
    'COUNT(transaction WHERE page = Cancel)',
    'MODE(transaction.session.userAgent)',
    'MODE(transaction.session.userId)'], axis=1, inplace=True)
    
    # one hot encoding
    cat_df = df_full.select_dtypes(include = 'object')
    cat_cols = cat_df.columns
    for feature in cat_df.columns:
        dfDummies = pd.get_dummies(cat_df[feature], prefix = feature)
        cat_df = pd.concat([cat_df, dfDummies], axis=1)

    # remove original cols
    cols_dummies = list(cat_df.columns)[len(cat_cols):]
    cat_df = cat_df[cols_dummies]
    df_full = pd.concat([df_full, cat_df], axis=1)
    df_full.drop(labels = cat_cols, axis = 1, inplace = True)
    df_full.drop(labels = ['gender_F'], axis = 1, inplace = True)
    
    # aggregrate positive, negative, neutral page counts
    df_full['positive_transaction_count'] = df_full['COUNT(transaction WHERE page = Upgrade)']+df_full['COUNT(transaction WHERE page = Add Friend)']+df_full['COUNT(transaction WHERE page = Thumbs Up)']+df_full['COUNT(transaction WHERE page = Add to Playlist)']
    df_full['negative_transaction_count'] = df_full['COUNT(transaction WHERE page = Thumbs Down)']+df_full['COUNT(transaction WHERE page = Cancellation Confirmation)']+df_full['COUNT(transaction WHERE page = Downgrade)']+df_full['COUNT(transaction WHERE page = Error)']
    df_full['neutral_transaction_count'] = df_full['COUNT(transaction WHERE page = Roll Advert)']+df_full['COUNT(transaction WHERE page = About)']+df_full['COUNT(transaction WHERE page = Home)']+df_full['COUNT(transaction WHERE page = Settings)']

    return df_full

def regroup_userAgent(x):
    if 'Macintosh' in x:
        return 'Macintosh'
    elif 'Windows' in x:
        return 'Windows'
    elif 'Linux' in x:
        return 'Linux'
    elif 'iPhone' in x:
        return 'iPhone'
    elif 'iPad' in x:
        return 'iPad'
    else:
        return 'others'

# function to automatically generate the features using dfs given data within the anchor period
def dfs_features(df, AGG_PRIMITIVES, unique_page_lst):
    es = ft.EntitySet(id="sparkify")
    es = es.entity_from_dataframe(
        entity_id="transaction", 
        dataframe=df, 
        index="transactionId", 
        time_index="transactionDate",
        variable_types={
                        "transactionId": ft.variable_types.Categorical,
                        "userId": ft.variable_types.Categorical,
                        "sessionKey": ft.variable_types.Categorical,
                       }
    )
    es = es.normalize_entity(
        base_entity_id="transaction",
        new_entity_id = "session",
        index="sessionKey",
        additional_variables=["userId","userAgent","gender","registrationDate"]
        )
    es = es.normalize_entity(
        base_entity_id="session",
        new_entity_id = "user",
        index="userId",
        additional_variables=["gender","registrationDate"]
        )
    es["transaction"]["page"].interesting_values = unique_page_lst
    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity="user",
                                     agg_primitives = AGG_PRIMITIVES)
    feature_matrix.reset_index(inplace=True)
    return feature_matrix

# functions to generate the labels
def get_label_cancel(userId):
    if userId not in labels['userId'].to_list():
        return 0
    else:
        return labels['is_cancel'][labels['userId']==userId].values[0]


def main():
    print("\nPreprocess")
    start = time.time()
    df = preprocess()
    print(f"  Number of rows after preprocessing = {df.count()}")
    print(f"  Time taken = {time.time() - start:.0f} s")
    
    print("\nSave data")
    df.to_csv('data/features.csv')

if __name__ == "__main__":
    main()
