import pandas as pd
df = pd.read_csv('/Users/liuzhen/AI-projects/gasifier-1d-kinetic/1d_kinetic_results.csv')
print(df[['z', 'T', 'CH4', 'O2', 'CO2', 'CO', 'H2O', 'H2']].head(5))
