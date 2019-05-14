import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df1 = pd.read_csv("./train_logs/loss1.csv", sep=';')
df1_2 = pd.read_csv("./train_logs/loss1_part2.csv", sep=';')
df1_3 = pd.read_csv("./train_logs/loss1_part3.csv", sep=';')
df2 = pd.read_csv("./train_logs/loss2.csv", sep=';')
df3 = pd.read_csv("./train_logs/loss3.csv", sep=';')

acc = np.array(df1['perceptual_acc'])
acc = np.concatenate([acc, np.array(df1_2['perceptual_acc'])])
acc = np.concatenate([acc, np.array(df1_3['perceptual_acc'])])

losses = np.array(df1['gen_loss'])
losses = np.concatenate([losses, np.array(df1_2['gen_loss'])])
losses = np.concatenate([losses, np.array(df1_3['gen_loss'])])

plt.subplot(2, 1, 1)
plt.title("Perceptual accuracy generatora")
plt.xlabel("epoki")
plt.ylabel("perceptual accuracy")
plt.plot(acc)

plt.subplot(2, 1, 2)
plt.title("Pełny loss generatora")
plt.xlabel("epoki")
plt.ylabel("perceptual accuracy")
plt.yscale("log")
plt.plot(losses)

plt.show()