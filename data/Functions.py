import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PowerTransformer

def data_cleaning(dataset):

    #Transforming Yes and No values into Booleans
    dataset = dataset.replace({ "No" : 0 , "Yes" : 1 })

    #Removal of policy_id column
    dataset = dataset.drop(columns = {"policy_id", 'Unnamed: 0'})

    # only considering rows where target variable is not nan
    dataset = dataset[(dataset.is_claim == 1) | (dataset.is_claim == 0)]

    dataset['torque_Nm'] = dataset['max_torque'].str.extract('(\d+)').astype(float)
    dataset['torque_rpm'] = dataset['max_torque'].str.extract('@(\d+)').astype(float)
    dataset['power_bhp'] = dataset['max_power'].str.extract('(\d+.\d+)').astype(float)
    dataset['power_rpm'] = dataset['max_power'].str.extract('@(\d+)').astype(float)

    # #Max Torque Dummies
    # torque_dummies = pd.get_dummies(dataset['max_torque'], prefix='torque')
    # dataset = pd.concat([dataset, torque_dummies], axis=1)

    # #Max Power Dummies
    # power_dummies = pd.get_dummies(dataset['max_power'], prefix='power')
    # dataset = pd.concat([dataset, power_dummies], axis=1)

    # columns = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera','is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_brake_assist','is_power_door_locks','is_central_locking','is_power_steering','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert']
    # for col in columns:
    #     dataset[col] = dataset[col].map({'Yes': 1, 'No': 0})
    dataset.drop(columns = {'max_torque','max_power'}, inplace=True)
    
    return dataset

def data_processing(dataset):

    dataset2 = data_cleaning(dataset)

    #Fuel Type
    fuel_type_dummies = pd.get_dummies(dataset2['fuel_type'], prefix='fuel_type')
    dataset2 = pd.concat([dataset2, fuel_type_dummies], axis=1)


    #Rear Brakes
    rear_brakes_dummies = pd.get_dummies(dataset2['rear_brakes_type'], prefix='rear_brakes')
    dataset2 = pd.concat([dataset2, rear_brakes_dummies], axis=1)


    #Transmission Type
    transmission_type_dummies = pd.get_dummies(dataset2['transmission_type'], prefix='transmission_type')
    dataset2 = pd.concat([dataset2, transmission_type_dummies], axis=1)

    #Segment
    segment_dummies = pd.get_dummies(dataset2['segment'], prefix='segment')
    dataset2 = pd.concat([dataset2, segment_dummies], axis=1)

    # steering_type
    steering_type_dummies = pd.get_dummies(dataset2['steering_type'], prefix='steering_type')
    dataset2 = pd.concat([dataset2, steering_type_dummies], axis=1)

    # engine_type
    engine_type_dummies = pd.get_dummies(dataset2['engine_type'], prefix='engine_type')
    dataset2 = pd.concat([dataset2, engine_type_dummies], axis=1)

    # Define regular expression pattern to match only digits
    pattern = r'\d+'

    # Apply regular expression to 'cluster' column and save as new column
    dataset2['area_cluster'] = dataset2['area_cluster'].apply(lambda x: re.search(pattern, x).group())
    dataset2['model'] = dataset2['model'].apply(lambda x: re.search(pattern, x).group())

    #Drop columns
    dataset2 = dataset2.drop(columns = {'fuel_type','rear_brakes_type','transmission_type','segment','steering_type','engine_type'}) # 'max_torque','max_power',

    dataset2['area_cluster'] = dataset2['area_cluster'].astype(int)
    dataset2['model'] = dataset2['model'].astype(int)

    # Powertransform relevant columns
    pt = PowerTransformer(method='yeo-johnson')
    dataset2.loc[:,['age_of_car','age_of_policyholder']] = pt.fit_transform(pd.DataFrame(dataset2.loc[:,['age_of_car','age_of_policyholder']]))

    dataset2 = dataset2.dropna()
    #Return dataset2

    return dataset2





