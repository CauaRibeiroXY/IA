import pandas as pd
import random as rd 
csv_input = pd.read_csv('card.csv')
#for i in len(csv_input):
    

csv_input['Escolaridade'] = [rd.randint(1,6) for i in range(0,len(csv_input))]

csv_input.to_csv('cardcredit.csv', index=False)


