from sklearn.neural_network import MLPClassifier
X = [[0,0],[0,1],[1,0],[1,1]]
y1 = [0,0,0,1]
y2 = [0,1,1,0]
clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5, 2),random_state=1)
clf.fit(X, y1)

print("Predictions for AND gate:", clf.predict(X))
print("Actual AND gate:", y1)

clf.fit(X, y2)
print("Predictions for AND gate:", clf.predict(X))
print("Actual AND gate:", y2)