import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix

from JointInv.machinelearn.base import load_disp

disp = load_disp()
plot_colors = "bry"
# only fit instantaneous period and group velocity
n_classes = 2
pair = [0,1]
x = disp.data[:, pair]
y = disp.target
plot_step = 0.02

# Train
#clf = DecisionTreeClassifier(min_samples_split=20).fit(x,y)
errweight = 1.0/disp.data[:,-1]
clf = DecisionTreeClassifier(min_samples_split=20).fit(x, y, sample_weight=errweight)

# plot the decision boundary
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.xlabel(disp.feature_names[pair[0]])
plt.ylabel(disp.feature_names[pair[1]])

# set limitation of x-axis
plt.xlim([25,100])
plt.ylim([2, 5])
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(x[idx,0], x[idx,1], c=color, label=disp.target_names[i], cmap=plt.cm.Paired)
plt.legend()
plt.xlabel("Period [s]")
plt.ylabel("Group Velocity [km/s]")
plt.savefig("Decision_surface.png")


# test model accuracy
disp_test = np.loadtxt("./data/test_acc.disp", delimiter=",")
# seperate features and targets
line, column = disp_test.shape
data = np.empty((line, column-1))
target = np.empty((line,),dtype=np.int)
data = disp_test[:,pair]
target = disp_test[:,-1]

# evaluate the separation model
## accuracy
acc = clf.score(data, target)
print("Accuracy={:8.5f}".format(acc))

## confusion matrix
all_predictions = clf.predict(data)
plt.matshow(confusion_matrix(target, all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('Confusion matrix of Decision Tree Model')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.savefig("confusion_matrix.png")
