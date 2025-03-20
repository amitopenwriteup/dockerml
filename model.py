# model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
