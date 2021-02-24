"Code of 19/02/2021 meant to read and filter AIS Spire data"

#%% Imports
import numpy as np
# import pickle
import pandas as pd
import time
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs

start_time = time.time()


#%% Loading File

filepath = r'C:/Users/Ruben/Documents/Thesis/Data/AIS/AISdata.csv'

data = pd.read_csv(filepath)

data.info()

#%% Adding time columns

time_column = data.loc[:, 'created_at']

hours = np.zeros(len(time_column))
date = np.zeros(len(time_column))
minutes = np.zeros(len(time_column))


for i in range(len(time_column)):
    if time_column[i][8] == 0:
        date[i] = int(time_column[i][9])
    else:
        date[i] = int(time_column[i][8:10])

    if time_column[i][11] == 0:
        hours[i] = int(time_column[i][12])
    else:
        hours[i] = int(time_column[i][11:13])
    if time_column[i][14] == 0:
        minutes[i] = int(time_column[i][15])
    else:
        minutes[i] = int(time_column[i][14:16])
    if i % 10000 == 0:
        print(i)

data['date'] = date
data['hours'] = hours
data['minutes'] = minutes

#Filter dataset by time

data = data.loc[data['hours'] > 9]
data = data.loc[data['hours'] < 14]


#%% Filtering File

#Dropping unneeded columns
# columns_to_drop = ['created_at', 'eta', 'destination', 'status', 'maneuver']
# data.drop(columns_to_drop, inplace=True, axis='columns')


#%% Sort by mmsi

data = data.sort_values(by=['mmsi', 'msg_type'], ascending=False)


#%% After filtering

headers = data.columns
data_head = data.head()


#%% Visualising

# fig = plt.figure(dpi=400)

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# ax.set_global()
# ax.scatter(lon,lat, s=4)

# plt.show()

#%% Final Message
headers = data.columns
data_head = data.head()

end_time = time.time()
execution_time = end_time - start_time
print(f'Done in {execution_time} seconds')
