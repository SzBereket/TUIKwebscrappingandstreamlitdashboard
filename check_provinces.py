import main
pt = main.load_population_timeseries()
provinces = pt['province'].unique()
print('Izmir present?', 'Izmir' in provinces)
print([p for p in provinces if p.startswith('I')][:10])
