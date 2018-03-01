from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
clf=tree.DecisionTreeClassifier()
s=SVC()
rf=RandomForestClassifier()
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
clf=clf.fit(X,Y)
s=s.fit(X,Y)
rf=rf.fit(X,Y)
prediction1=clf.predict([ [160, 60, 38]])
prediction2=s.predict([ [160, 60, 38]])
prediction3=rf.predict([ [160, 60, 38]])
print ("Tree={0}\nSVC={1}\nForest{2}".format(prediction1,prediction2,prediction3))
print (accuracy_score(prediction1,prediction2,prediction3))
