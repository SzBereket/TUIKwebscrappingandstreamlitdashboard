import main
single = main.load_single_age_population()
print(single['sex'].unique()[:5])
print(single[single['sex']=='Toplam'].head())
