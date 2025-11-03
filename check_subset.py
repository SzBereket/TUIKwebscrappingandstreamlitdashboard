import main
pt = main.load_population_timeseries()
print(pt[pt['province']=='Izmir'].head())
print(pt[pt['province']=='Turkiye'].head())
