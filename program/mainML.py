import csv
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, StratifiedKFold

# Random Forest
from sklearn import ensemble, metrics, preprocessing


#print(file_name)

date = []
aveTmp = []
highestTmp = []
lowestTmp = []
useAmomunt = []
X = []
X_train = []

#電力使用量を取り込む
file_name = os.listdir('../data/Electrical/')
for name in file_name:
    firstloop = True
    name = '../data/Electrical/' + name
    with open(name, encoding="shift_jis") as csv_file:
        f = csv.reader(csv_file, delimiter=",")
        for j in f:
            if firstloop:
                firstloop = False
                continue
            date.append(j[2])
            useAmomunt.append(j[3])
        #print(date)
        #print(useAmomunt)

#気温データを取り込む
file_name = os.listdir('../data/weather/')
for name in file_name:
        roopCounter = 0
        name = '../data/weather/' + name
        with open(name, encoding="shift_jis") as csv_file:
            f = csv.reader(csv_file, delimiter=",")
            for j in f:
                if roopCounter < 6:
                    roopCounter+=1
                    continue
                aveTmp.append(j[1])
                highestTmp.append(j[4])
                lowestTmp.append(j[7])
        print(aveTmp)
for i in range(len(aveTmp)):
    X.append([aveTmp[i],highestTmp[i],lowestTmp[i]])
print(X)
print(useAmomunt)


sscaler = preprocessing.StandardScaler()
X_std = sscaler.fit_transform(X)
X = X_std

X_train, X_test, Y_train, Y_test = train_test_split(X, useAmomunt, test_size=0.5, random_state=0)

model = ensemble.RandomForestClassifier()
model.fit(X_train,Y_train)
Y_predict = model.predict(X_test)
print(Y_predict)
print(metrics.classification_report(Y_test,Y_predict))