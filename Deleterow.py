import pandas as pd
import random as rd 
df = pd.read_csv('cardcredit.csv')
print(df)
df = df.drop(columns=['indice'])
print(df)
df.to_csv('cardcredit.csv', index=False)
