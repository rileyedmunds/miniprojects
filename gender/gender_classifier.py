from sklearn import tree, svm, neighbors, naive_bayes

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf_tree = tree.DecisionTreeClassifier()
clf_kneighbors = neighbors.KNeighborsClassifier()
clf_svm = svm.SVC()
clf_bayes = naive_bayes.GaussianNB()


clf_tree = clf_tree.fit(X,Y)
clf_kneighbors = clf_kneighbors.fit(X,Y)
clf_svm = clf_svm.fit(X,Y)
clf_bayes = clf_bayes.fit(X,Y)


sample = ([190,70,43])

out_tree = clf_tree.predict(sample)
out_kneighbors = clf_kneighbors.predict(sample)
out_svm = clf_svm.predict(sample)
out_bayes = clf_bayes.predict(sample)

print out_tree
print out_kneighbors
print out_svm
print out_bayes