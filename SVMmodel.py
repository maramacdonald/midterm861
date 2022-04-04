import kfold_template
import pandas 

from sklearn import svm 


dataset = pandas.read_csv("survey_dataset.csv")
dataset = dataset.drop('horoscope', axis=1)
baddata = dataset[dataset['age'] <= 0].index
dataset = dataset.drop(baddata , inplace=False) 

dataset.game.factorize()
dataset['gamecode'] = dataset.game.factorize()[0]

dataset = dataset.drop('gender', axis=1)
dataset = dataset.drop('region', axis=1)
dataset = dataset.drop('personality8', axis=1)
dataset = dataset.drop('game', axis=1)



target = dataset.iloc[:,10].values
data = dataset.iloc[:,0:10].values


r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, 
																		target, 
																		4, 
																		svm.SVC(kernel="linear"),
																		 1,
																		  1)
print(r2_scores)
print(accuracy_scores)
for i in confusion_matrices:
	print(i)

