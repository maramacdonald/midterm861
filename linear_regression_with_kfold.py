import pandas
from sklearn import linear_model

from sklearn.model_selection import KFold

from sklearn import metrics

from matplotlib import pyplot

import numpy

dataset = pandas.read_csv("survey_dataset.csv")

dataset = dataset.drop('horoscope', axis=1)
baddata = dataset[dataset['age'] <= 0].index
dataset = dataset.drop(baddata , inplace=False) 
dataset = dataset.drop('gender', axis=1)
dataset = dataset.drop('region', axis=1)
dataset = dataset.drop('personality8', axis=1)
dataset.game.factorize()
dataset['gamecode'] = dataset.game.factorize()[0]
dataset = dataset.drop('game', axis=1)


target = dataset.iloc[:,10].values
data = dataset.iloc[:,0:10].values

kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)


i=0
for training_index, test_index in kfold_object.split(data):
	i=i+1
	print("Round: ", str(i))
	print("Training Index:  ")
	print(training_index)
	print("Test Index:  ")
	print (test_index)
	print ("\n\n")
	data_training = data[training_index]
	target_training = target[training_index]
	data_test = data[test_index]
	target_test = target[test_index]
	machine = linear_model.LinearRegression()
	machine.fit(data_training, target_training)
	print(machine.score(data_training, target_training))
	new_target = machine.predict(data_test)
	print(metrics.r2_score(target_test, new_target))




