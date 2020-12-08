"Code of 7/12/2020 meant as test to read AIS Spire data"

#Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import cartopy.crs as ccrs

filepath = r'C:/Users/Ruben/Documents/Thesis/Data/AIS-spire/globaloct24_1.csv'


start_time = time.time()

data = pd.read_csv(filepath)

print(data.head())

test_data = data.head()

headers = data.columns


lat = test_data['latitude']
lon = test_data['longitude']

print(headers[0])





# test_data.drop(to_drop, inplace = True, axis = 'columns')

fig = plt.figure(dpi = 400)

ax = plt.axes(projection = ccrs.PlateCarree())
ax.coastlines()
ax.set_global()
ax.scatter(lon,lat, s=4)



plt.show()


print(f'Execution took {np.abs(start_time - time.time()):.2f} seconds')