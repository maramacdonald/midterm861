README.md

IMPORT SURVEY DATA 
dataset = pandas.read_csv("survey_dataset.csv")

CHECK DATA VARIABLES AND VALUES
print(dataset.describe())

DROP IRRELEVENT CHARACTERISTICS OR PROBLEMATIC OBSERVATIONS
dataset = dataset.drop('horoscope', axis=1)
baddata = dataset[dataset['age'] <= 0].index
dataset = dataset.drop(baddata , inplace=False) 

TRANSFORM VIDEO GAME TITLES INTO UNIQUE NUMERICAL VALUES
dataset.game.factorize()

CREATE NEW COLUMN FOR UNIQUE VIDEO GAME TITLE CODES SO IT CAN BE USED
dataset1['gamecode'] = dataset1.game.factorize()[0]


MAKE MODEL SIMPLER: USE A SMALLER SUBSET OF CUSTOMER CHARACTERISTICS 
dataset = dataset.drop('gender', axis=1)
dataset = dataset.drop('region', axis=1)
dataset = dataset.drop('personality8', axis=1)
dataset = dataset.drop('game', axis=1)
CHECK DATA VARIABLES AND VALUES
print(dataset.describe())
print(dataset)

OUR TARGET IS THE GAMECODES BECAUSE WE ARE TRYING TO PREDICT CUSTOMER'S FAVORITE GAMES
target = dataset.iloc[:,10].values
data = dataset.iloc[:,0:10].values


SPLIT OUR DATA UP INTO 4 EVEN RANDOM GROUPS SO WE CAN RUN 4 DIFFERENT TESTS
kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)

TRY OUT DIFFERENT MODELS

machine = linear_model.LinearRegression()


machine = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000000)


