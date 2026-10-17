import lasio
las = lasio.read("sample.las")
df = las.df()
df.head()
