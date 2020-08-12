from sklearn import tree
import matplotlib .pyplot as plt

X_names = ['peso', 'textura']
X = [
  [150, 0],
  [170, 0],
  [140, 1],
  [130, 1],
  [90, 0],
  [80, 0],
  [100, 1],
  [105, 1]
]

y_names = ['laranja', 'maça', 'limão', 'banana']
y = [0, 0, 1, 1, 2, 2, 3, 3]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

fig = plt.figure()

tree.plot_tree(
  clf,
  feature_names=X_names,
  class_names = y_names,
  filled=True,
  impurity=False,
  rounded=True
  )

plt.savefig('frutas.png')
