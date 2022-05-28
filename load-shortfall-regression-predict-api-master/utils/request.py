"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.

"""

# Import dependencies
import requests
import pandas as pd
import numpy as np

# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Kaggle challenge.
test = pd.read_csv('./data/df_test.csv')

# Feature Engineering

## Replacing null values using mean
test['Valencia_pressure'].fillna(test['Valencia_pressure'].mean(), inplace=True)

# # drop unnecessary variable, and those object data type variables
test_drop=test.drop('Unnamed: 0', axis=1)
df_drop2=test_drop.drop('Valencia_wind_deg', axis=1)
df_drop3=df_drop2.drop('Seville_pressure', axis=1)

# Melt the data
df_melt = pd.melt(
    df_drop3,
    id_vars=['time'],
    var_name='weather',
    value_name='value'
)

# Make a new column [data] in the data which is going to be easy to select the require weather features for a new engineered file
df_melt['data']=df_melt['weather'].str.replace('_',' ')

#Create individual files for the weather features which are later going to be joined to make a spain_trained file
wind_speed_df=df_melt[df_melt.data.str.contains('wind speed')]
wind_deg_df=df_melt[df_melt.data.str.contains('wind deg')]
humidity_df=df_melt[df_melt.data.str.contains('humidity')]
clouds_all_df=df_melt[df_melt.data.str.contains('clouds all')]
pressure_df=df_melt[df_melt.data.str.contains('pressure')]
snow_3h_df=df_melt[df_melt.data.str.contains('snow 3h')]
rain_3h_df=df_melt[df_melt.data.str.contains('rain 3h')]
temp_max_df=df_melt[df_melt.data.str.contains('temp max')]
temp_min_df=df_melt[df_melt.data.str.contains('temp min')]

# Making a new column with the required values for each weather features.
wind_speed_df['wind_speed']=wind_speed_df['value']
wind_deg_df['wind_deg']=wind_deg_df['value']
humidity_df['humidity']=humidity_df['value']
clouds_all_df['clouds_all']=clouds_all_df['value']
pressure_df['pressure']=pressure_df['value']
snow_3h_df['snow_3h']=snow_3h_df['value']
rain_3h_df['rain_3h']=rain_3h_df['value']
temp_max_df['temp_max']=temp_max_df['value']
temp_min_df['temp_min']=temp_min_df['value']

# Create a city column in the individual spain weather data

variable_split = wind_speed_df['weather'].str.split('_')
wind_speed_df['city'] = variable_split.str.get(0)
variable_split2 = wind_deg_df['weather'].str.split('_')
wind_deg_df['city'] = variable_split2.str.get(0)
variable_split3 = humidity_df['weather'].str.split('_')
humidity_df['city'] = variable_split3.str.get(0)
variable_split4 = clouds_all_df['weather'].str.split('_')
clouds_all_df['city'] = variable_split4.str.get(0)
variable_split5 = pressure_df['weather'].str.split('_')
pressure_df['city'] = variable_split5.str.get(0)
variable_split6 = snow_3h_df['weather'].str.split('_')
snow_3h_df['city'] = variable_split6.str.get(0)
variable_split7 = rain_3h_df['weather'].str.split('_')
rain_3h_df['city'] = variable_split7.str.get(0)
variable_split8 = temp_max_df['weather'].str.split('_')
temp_max_df['city'] = variable_split8.str.get(0)
variable_split10 = temp_min_df['weather'].str.split('_')
temp_min_df['city'] = variable_split10.str.get(0)

# Select the required parameters to newly formed feature engineered files

new_wind_speed_df=wind_speed_df[['time', 'city', 'wind_speed']]
new_wind_deg_df=wind_deg_df[['time', 'city', 'wind_deg']]
new_humidity_df=humidity_df[['time', 'city', 'humidity']]
new_clouds_all_df=clouds_all_df[['time', 'city', 'clouds_all']]
new_pressure_df=pressure_df[['time', 'city', 'pressure']]
new_snow_3h_df=snow_3h_df[['time', 'city', 'snow_3h']]
new_rain_3h_df=rain_3h_df[['time', 'city', 'rain_3h']]
new_temp_max_df=temp_max_df[['time', 'city', 'temp_max']]
new_temp_min_df=temp_min_df[['time', 'city', 'temp_min']]

# Merge using outer join to collect all the information from all the files. These create a new weather file for spain with all the data filtered well with all the cities on their own column

merge1=pd.merge(new_wind_speed_df,new_wind_deg_df, how='outer')
merge2=pd.merge(merge1,new_humidity_df, how='outer')
merge3=pd.merge(merge2,new_clouds_all_df, how='outer')
merge4=pd.merge(merge3,new_pressure_df, how='outer')
merge5=pd.merge(merge4,new_snow_3h_df, how='outer')
merge6=pd.merge(merge5,new_rain_3h_df, how='outer')
merge7=pd.merge(merge6,new_temp_max_df, how='outer')
newspain_df=pd.merge(merge7,new_temp_min_df, how='outer')
newspain_df

newspaindf_grp=newspain_df.groupby(['time'])

spain_test=newspaindf_grp[['wind_speed', 'wind_deg', 'humidity', 'clouds_all', 'pressure', 'temp_max', 'temp_min', 'snow_3h', 'rain_3h' ]].mean().reset_index()

#Engineering New Features ( i.e Desampling the Time) that will help us in our modeling
spain_test['Year']  = spain_test['time'].astype('datetime64').dt.year
spain_test['Month_of_year']  = spain_test['time'].astype('datetime64').dt.month
spain_test['Week_of_year'] = spain_test['time'].astype('datetime64').dt.weekofyear
spain_test['Day_of_year']  = spain_test['time'].astype('datetime64').dt.dayofyear
spain_test['Day_of_month']  = spain_test['time'].astype('datetime64').dt.day
spain_test['Day_of_week'] = spain_test['time'].astype('datetime64').dt.dayofweek
spain_test['Hour_of_week'] = ((spain_test['time'].astype('datetime64').dt.dayofweek) * 24 + 24) - (24 - spain_test['time'].astype('datetime64').dt.hour)
spain_test['Hour_of_day']  = spain_test['time'].astype('datetime64').dt.hour

#spain_test_drop=spain_test.drop('time', axis=1)




# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = spain_test.iloc[1].to_json()
print(feature_vector_json)

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://127.0.0.1:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {test.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()[0]}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)
