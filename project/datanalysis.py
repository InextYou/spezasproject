import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("../data/Component_Faults_Data.csv")

print('----------whole dataset----------')
print(df.describe())

headers = list(df)
print(headers)
for header in headers:
    if header != 'class':
        print('----------'+header+'----------')
        print(df[header].describe())
        df[header].hist(bins=df[header].size) ##macht des sinn? 100 oder 1000 einträge im Histogramm bei 28k insgesamt isch ja ned die ganze wahrheit
        plt.show() ##todo: scaling von der axe isch manchmal verschickt



##to do: plot min and max values of all cols to show distribution

df.max().plot()
plt.show()
df.min().plot()
plt.show()

df.max.plot()
df.min.plot()
plt.show()
#todo: 0 y achse nach oben verschieben, scaling andersch dass 0.xxx werte auch angezeigt werden?




