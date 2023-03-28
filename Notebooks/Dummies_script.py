#Load data

import pandas as pd

data = pd.read_csv("INSERT PATH")

#Binary Variables

columns = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera','is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_brake_assist','is_power_door_locks','is_central_locking','is_power_steering','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert']
for col in columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

#Dummies

torque_dummies = pd.get_dummies(data['max_torque'], prefix='torque')
data = pd.concat([data, torque_dummies], axis=1)

power_dummies = pd.get_dummies(data['max_power'], prefix='power')
data = pd.concat([data, power_dummies], axis=1)

#Fuel Type
fuel_type_dummies = pd.get_dummies(data['fuel_type'], prefix='fuel_type')
data = pd.concat([data, fuel_type_dummies], axis=1)

#Rear Brakes
rear_brakes_dummies = pd.get_dummies(data['rear_brakes_type'], prefix='rear_brakes')
data = pd.concat([data, rear_brakes_dummies], axis=1)

#Transmission Type
transmission_type_dummies = pd.get_dummies(data['transmission_type'], prefix='transmission_type')
data = pd.concat([data, transmission_type_dummies], axis=1)

#Save

data.to_csv('cars_modified.csv', index=False)