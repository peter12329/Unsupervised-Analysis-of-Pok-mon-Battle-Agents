import pandas as pd

df = pd.read_csv("battles.csv")

print(df.head())

print("\nInfo:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nAverage Battle Length:")
print(df["battle_length"].mean())

print("\nShortest Battle:")
print(df["battle_length"].min())

print("\nLongest Battle:")
print(df["battle_length"].max())

print("\nPlayer 1 Wins:")
print((df["player1_reward"] == 1).sum())

print("\nPlayer 2 Wins:")
print((df["player2_reward"] == 1).sum())