"Code of 7/12/2020 meant as test to read AIS Spire data"

#Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

filepath = r'C:/Users/Ruben/Documents/Thesis/Data/AIS-spire/globaloct24_1.csv'


start_time = time.time()


data = pd.read_csv(filepath)



print(data.head())

print(f'Execution took {start_time - time.time()} seconds')