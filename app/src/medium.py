import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('./SapphiroJeans.csv')
df = df.select_dtypes(exclude=['object'])

# Store variables target y and the first two features as X (waist length and hips length of the jeans)
X = np.asarray(df.drop(['currentSize', 'Leg', 'Shape'], axis=1))  # Waist and Hips sizes
y = np.asarray(df['currentSize'])  # Class (size)

# Split the dataset in train and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)

# step size in the mesh, it alters the accuracy of the plotprint
# to better understand it, just play with the value, change it and print it
h = .02

# create the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# create the title that will be shown on the plot
titles = ['Linear kernel', 'RBF kernel', 'Polynomial kernel', 'Sigmoid kernel']

score = linear.score(X_test, y_test)
# defines how many plots: 2 rows, 2columns=> leading to 4 plots
plt.subplot(2, 2, 1)  # i+1 is the index
# space between plots
plt.subplots_adjust(wspace=0.4, hspace=0.4)
Z = linear.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap=plt.cm.viridis,
    edgecolors="k",
    alpha=0.6,
)

plt.xlabel('Waist length')
plt.ylabel('Hips width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.text(
    xx.max() - 0.3,
    yy.min() + 0.3,
    ("%.2f" % score).lstrip("0"),
    size=15,
    horizontalalignment="right",
)
plt.title('Linear Kernel')
plt.tight_layout()
plt.show()

linear_pred = linear.predict(X_test)
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
sig_pred = sig.predict(X_test)

# retrieve the accuracy and print it for all 4 kernel functions
accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)
print('Accuracy Linear Kernel:', accuracy_lin)
print('Accuracy Polynomial Kernel:', accuracy_poly)
print('Accuracy Radial Basis Kernel:', accuracy_rbf)
print('Accuracy Sigmoid Kernel:', accuracy_sig)
