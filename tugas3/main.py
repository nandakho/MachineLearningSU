# Naive Bayes algorithm dengan ekstra komen
# Code by: Ananda Khosuri & Rivan Oktavianus
# import library
import pandas as pd
import numpy as np


# function rumus hitung, argumen input: nilai yang diprediksi, s: standard deviation, m: mean
def rumus(inputs, s, m):
    result = 1 / (s * np.sqrt(2 * np.pi)) * pow(np.e, (-1 * (pow(inputs - m, 2)) / (2 * (pow(s, 2)))))
    return result


# jika mau debug ubah ke 1
dbg = 0

# import dataset di sini
dataset = pd.read_csv("data.csv")

# set kriteria dari header csv
kriteria = dataset.columns

# set goal dari header paling kanan
goal = kriteria[kriteria.size - 1]
goal_unik = dataset[goal].unique()

# set mean dan std deviasi
mean = dataset.groupby(goal).mean()
std = dataset.groupby(goal).std()

# set jumlah data
n_max = dataset.shape[0]
n = [0] * len(goal_unik)
for x in range(len(goal_unik)):
    n[x] = dataset.groupby(goal).size()[goal_unik[x]] / n_max

# use for debug only
if dbg == 1:
    for i in range(len(kriteria) - 1):
        print(kriteria[i])
        print("Mean:", mean[kriteria[i]])
        print("Std:", std[kriteria[i]])

# print menu untuk input data yang akan diprediksi
print("Insert your data for prediction")

# init array inp dan loop untuk input
inp = [0] * (len(kriteria) - 1)
for k in range(len(kriteria) - 1):
    inp_prompt = "Insert " + kriteria[k] + ": "
    inp[k] = float(input(inp_prompt))
    # loop untuk menyimpan result sementara
    for i in range(len(goal_unik)):
        n[i] *= rumus(inp[k], std[kriteria[k]][goal_unik[i]], mean[kriteria[k]][goal_unik[i]])

# debug tampilkan hasil kalkulasi rumus
if dbg == 1:
    for i in range(len(goal_unik)):
        print(goal_unik[i], ":", n[i])

print("Keputusannya", goal, ":", goal_unik[np.argmax(n)])
