import pandas as pd
import quandl
#&api_key=6aXkBG3GyC8hJNgGUGqK
df=quandl.get("SSE/GGQ1")
print(df.head())
