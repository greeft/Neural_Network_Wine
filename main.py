import pandas as pd
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("wine.csv")
#print(data.head())
le = preprocessing.LabelEncoder()
scaler = StandardScaler()
data.describe().transpose()
#print(data.shape)
Type = data["Typ"]


alcohol = data["Alcohol"]
alcohol = np.int64(np.around(alcohol * 10 ** 3))

Malic_acid = data["Malic_acid"]
Malic_acid = np.int64(np.around(Malic_acid * 10 ** 3))

Ash = data["Ash"]
Ash = np.int64(np.around(Ash * 10 ** 3))

Alcalinity_of_ash = data["Alcalinity_of_ash"]
Alcalinity_of_ash = np.int64(np.around(Alcalinity_of_ash * 10 ** 3))

Magnesium = data["Magnesium"]
Magnesium = np.int64(np.around(Magnesium * 10 ** 3))

Total_phenols = data["Total_phenols"]
Total_phenols = np.int64(np.around(Total_phenols * 10 ** 3))

Flavanoids = data["Flavanoids"]
Flavanoids = np.int64(np.around(Flavanoids * 10 ** 3))

Nonflavanoid_phenols = data["Nonflavanoid_phenols"]
Nonflavanoid_phenols = np.int64(np.around(Nonflavanoid_phenols * 10 ** 3))

Proanthocyanins = data["Proanthocyanins"]
Proanthocyanins = np.int64(np.around(Proanthocyanins * 10 ** 3))

Color_intensity = data["Color_intensity"]
Color_intensity = np.int64(np.around(Color_intensity * 10 ** 3))

Hue = data["Hue"]
Hue = np.int64(np.around(Hue * 10 ** 3))

protein_concentration = data["protein_concentration"]
protein_concentration = np.int64(np.around(protein_concentration * 10 ** 3))

Proline = data["Proline"]
y_list = np.array(data["Typ"])
x_list =list(zip(alcohol, Malic_acid, Ash, Alcalinity_of_ash, Magnesium, Total_phenols, Flavanoids, Nonflavanoid_phenols,
        Proanthocyanins, Color_intensity, Hue, protein_concentration, Proline))
#print(y_list)
#print(x_list)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_list, y_list, test_size=0.1)
# fit(x_list,y_list)
#print('essa')
mlp = MLPClassifier(hidden_layer_sizes=(26,13,), activation='tanh', solver='lbfgs', max_iter=500)
mlp.fit(x_train,y_train)
acc = mlp.score(x_test,y_test)
print(acc)
predictions = mlp.predict(x_test)
for i in range (len(predictions)):
    print(predictions[i],x_test[i],y_test[i])
#predction_score = Neural_Network.score(x_test, y_test)

#print(predction_score)
