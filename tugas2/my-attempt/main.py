from pandas import read_csv

kriteria = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
dataset = read_csv('iris.csv')  # csv name
category = dataset['variety'].unique()
test = dataset.groupby('variety')
inp = [0] * len(kriteria)
jenis = [0] * len(category)

print("Insert your data for prediction")
for k in range(len(kriteria)):
    inp_prompt = "Insert " + kriteria[k] + ": "
    inp[k] = float(input(inp_prompt))
print("")

for c in range(len(category)):
    # print(category[c])
    for k in range(len(kriteria)):
        # print(kriteria[k], ":", test[kriteria[k]].min()[c], "<=", inp[k], "<=", test[kriteria[k]].max()[c])
        if test[kriteria[k]].min()[c] <= inp[k] <= test[kriteria[k]].max()[c]:
            jenis[c] += 1
            # print("true")
    # print("")

# print(jenis)

print("Result:")
for j in range(len(jenis)):
    print(jenis[j] / len(kriteria) * 100, "%", category[j])

print("Your iris is a", category[jenis.index(max(jenis))], "variant")
