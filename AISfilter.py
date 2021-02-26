"Code of 19/02/2021 meant to read and filter AIS Spire data"

#%% Imports
import numpy as np
import pickle
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

#%% Filtering dataset

#Filter dataset by time

data = data.loc[(data.hours > 9) & (data.hours < 14)]

# Filter dataset by cargo type

# data = data.loc[data['ship_and_cargo_type'] > 69]
# data = data.loc[data['ship_and_cargo_type'] < 90]
#%% Sort by mmsi

data = data.sort_values(by=['mmsi', 'msg_type'])

# Collect static ship data

ship_data = data.loc[(data['msg_type'] == 5) | (data['msg_type'] == 24)]
ship_data = ship_data.drop_duplicates(subset=['mmsi'])

location_data = data.loc[~((data['msg_type'] == 5) | (data['msg_type'] == 24))]
df = location_data

# ship_data = ship_data[:4]

#%% Taking together 2 datasets
# Iterate over mmsi in ship data:

for i, row in ship_data.iterrows():
    curr_mmsi = row.mmsi
    curr_name = row['name']
    curr_draught = row.draught
    curr_ship_and_cargo_type = row.ship_and_cargo_type
    curr_length = row.length
    curr_width = row.width
    # print(row)
    print(curr_mmsi)
      
    df.loc[df['mmsi'] == curr_mmsi, 'name'] = str(curr_name)
    df.loc[df['mmsi'] == curr_mmsi, 'draught'] = str(curr_draught)
    df.loc[df['mmsi'] == curr_mmsi, 'ship_and_cargo_type'] = str(curr_ship_and_cargo_type)
    df.loc[df['mmsi'] == curr_mmsi, 'length'] = str(curr_length)
    df.loc[df['mmsi'] == curr_mmsi, 'width'] = str(curr_width)
    
# Copy relevant data to variables (mmsi, name, draught, ship_and_cargo_type, length, width)
# Fill location data at this mmsi with these variables

#%% After filtering

headers = data.columns
data_head = data.head()

# for index, row in data.iterrows():
#     print(row)


#%% Filtering and saving

df = df[df['name'].notna()].sort_values(by=['name'], ascending=False)
df = df[df['name'] != 'nan']

savepath = r'C://Users/Ruben/Documents/Thesis/Data/AIS/AISdataframe.p'

pickle.dump(df, open(savepath, "wb"))

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
