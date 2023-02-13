import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve, cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree._tree import TREE_LEAF

# Source of most code is from sklearn user guide:



# function to plot mean accuracy v.s. iterations
def plot_iterations(solver, X, y, niters, title, fname, folder):
    tr_scores = np.zeros((len(niters), 5), dtype=np.float32)
    tt_scores = np.zeros((len(niters), 5), dtype=np.float32)

    for j, i in enumerate(niters):
        cv_iter = KFold(n_splits=5, shuffle=True).split(X)
        result = np.zeros((2, 5), dtype=np.float32)
        for n, iter in enumerate(cv_iter):
            solver.max_iter = i
            solver.fit(X[iter[0]], y[iter[0]])
            result[:, n] = np.array([solver.score(X[iter[0]], y[iter[0]]), solver.score(X[iter[1]], y[iter[1]])])

        tr_scores[j, :] = result[0, :]
        tt_scores[j, :] = result[1, :]

    tr_means = np.mean(tr_scores, axis=1)
    tr_stdev = np.std(tr_scores, axis=1)
    tt_means = np.mean(tt_scores, axis=1)
    tt_stdev = np.std(tt_scores, axis=1)

    plt.figure()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean Accuracy Score')
    plt.title(title)

    plt.fill_between(niters, tr_means-tr_stdev, tr_means+tr_stdev, color='darkorange', alpha=0.1)
    plt.fill_between(niters, tt_means-tt_stdev, tt_means+tt_stdev, color='navy', alpha=0.1)
    plt.plot(niters, tr_means, 'o-', color='darkorange', label='train')
    plt.plot(niters, tt_means, 'o-', color='navy', label='test')
    plt.legend(loc='best')

    plt.savefig(folder + fname, format='png')
    plt.close()


# function to plot cross-validated learning curves
def plot_learning_curve(solver, X, y, title, fname, folder):
    full_name = folder + fname
    train_szs, train_scores, test_scores = learning_curve(solver, X, y, train_sizes=np.linspace(0.2, 1.0, 5))

    tr_means = np.mean(train_scores, axis=1)
    tr_stdev = np.std(train_scores, axis=1)
    tt_means = np.mean(test_scores, axis=1)
    tt_stdev = np.std(test_scores, axis=1)
    norm_train_szs = (train_szs / max(train_szs)) * 100.0

    plt.figure()
    plt.title(title)
    plt.xlabel('% of Training Data')
    plt.ylabel('Mean Accuracy Score')


    plt.plot(norm_train_szs, tr_means, 'o-', color='darkorange', label='train')
    plt.plot(norm_train_szs, tt_means, 'o-', color='navy', label='test')
    plt.fill_between(norm_train_szs, tr_means-tr_stdev, tr_means+tr_stdev, color='darkorange', alpha=0.1)
    plt.fill_between(norm_train_szs, tt_means-tt_stdev, tt_means+tt_stdev, color='navy', alpha=0.1)
    plt.legend(loc='best')

    plt.savefig(full_name, format='png')
    plt.close()


# Prune DT based on given threshold
def prune_index(inner_tree, index, threshold):
    # https://stackoverflow.com/questions/49428469/pruning-decision-trees
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


def decision_tree(X, y, folder, f):
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    plt.figure()
    plt.title('DT v.s. alphas')
    plt.xlabel('Alpha')
    plt.ylabel('Leaf Impurity')
    plt.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    plt.savefig(folder + 'Alpha_vs_impurty.png', format='png')
    plt.close()

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        t0 = time.time()
        clf.fit(X_train, y_train)
        print(f'Tree fitting time: alpha={ccp_alpha} ' + str(time.time() - t0), file=f)
        clfs.append(clf)

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    plt.figure()
    plt.title('DT Accuracy of Alphas')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    plt.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    plt.savefig(folder + 'tree_Alpha_vs_accuracy.png', format='png')
    plt.legend()
    plt.close()

    plot_learning_curve(clf, X, y, "tree_learning_curve", "tree_learning_curve.png", folder)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    print('\nMax Depth Hyperparameter Tuning', file=f)

    depth_clfs = []

    for i in range(1, 10):
        clf = DecisionTreeClassifier(max_depth=i, random_state=0)
        t0 = time.time()
        clf.fit(X_train, y_train)
        print(f'Tree fitting time: depth={i} ' + str(time.time() - t0), file=f)
        depth_clfs.append((i, clf))

    best_acc = -10000
    best_depth = -1000
    depths = []
    accs = []

    for clf in depth_clfs:
        ac = accuracy_score(y_test, clf[1].predict(X_test))
        depths.append(clf[0])
        accs.append(ac)
        if ac >= best_acc:
            best_depth = clf[0]
            best_acc = ac

    plt.figure()
    plt.plot(depths, accs)
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Max Depth vs Accuracy")
    plt.savefig(folder + "tree_Depth_vs_Accuracy", format='png')
    plt.close()

    print(f'Best Depth: {best_depth}', file=f)
    print(f'Best Accuracy: {best_acc}', file=f)


# simple MLP neural network that experiments on multiple layers, alphas, and activation functions
def neural_networks(X, y, folder, f):
    # Neural Network
    print('\n=== Neural Networks ===', file=f)

    # varying node + layers count (X, Y)
    print('\nExperiment 1: Nodes & Layers', file=f)
    layer_1 = [5, 10, 15, 20, 30, 50, 75, 100]
    layer_2 = [2, 3, 5]
    for l1 in layer_1:
        for l2 in layer_2:
            network = (l1, l2)
            clf_nn = MLPClassifier(network, max_iter=200, activation='logistic', solver='lbfgs', alpha=0.1)
            t0 = time.time()
            print(f"{network}: " + str(np.mean(cross_val_score(clf_nn, X, y))), file=f)
            print('runtime:' + str(time.time() - t0), file=f)


    # varying alphas
    print('\nExperiment 2: Alphas', file=f)
    alphas = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.32, 0.6, 0.8, 1.0, 2.0, 3.16, 5.0, 10.0, 100.0]
    result = np.zeros((len(alphas), 5), dtype=np.float32)

    for i, a in enumerate(alphas):
        clf_nn = MLPClassifier((5, 2), max_iter=200, activation='logistic', solver='lbfgs', alpha=a)
        t0 = time.time()
        result[i, :] = cross_val_score(clf_nn, X, y)
        print(f'runtime: alpha={a} ' + str(time.time() - t0), file=f)

    print(result, file=f)

    nn_cv_mean = np.mean(result, axis=1)
    nn_cv_stds = np.std(result, axis=1)

    plt.figure()
    plt.title('Neural Network v.s. alphas')
    plt.xscale("log")
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.fill_between(alphas, nn_cv_mean-nn_cv_stds, nn_cv_mean+nn_cv_stds, color='darkorange', alpha=0.1)
    plt.plot(alphas, nn_cv_mean, 'o-', color='darkorange')
    plt.savefig(folder + 'nn_vs_alphas.png', format='png')
    plt.close()


    # Activation functions
    print('\nExperiment 3: Activation Functions', file=f)

    activation = ["logistic", "tanh", "relu"]
    result = np.zeros((len(activation), 5), dtype=np.float32)

    for i, func in enumerate(activation):
        clf_nn = MLPClassifier((5, 2), max_iter=1000, activation=func, solver='lbfgs', alpha=0.1)
        t0 = time.time()
        plot_learning_curve(clf_nn, X, y, 'neural network', f'nn_{func}_sklearn_learning_curve.png', folder)
        result[i, :] = cross_val_score(clf_nn, X, y)
        print(f'5-fold CV for NN: func={func} ' + str(time.time() - t0), file=f)

    print(result, file=f)


def boosting_tree(X, y, folder, f):
    # Ada boosting ensemble method

    print('\nADA Boosted Tree', file=f)


    print('\nADA Graph Pruning', file=f)
    clf = DecisionTreeClassifier()
    clf_ada = AdaBoostClassifier(clf)
    plot_learning_curve(clf_ada, X, y,'Ada Boosted Tree (before pruning)', 'tree_ada_before_sklearn_learning_curve.png', folder)

    ada_pre_pruning = np.zeros((5, 20), dtype=np.float32)
    t0 = time.time()
    for i, m in enumerate(range(1, 10, 2)):
        for j, n in enumerate(range(2, 160, 30)):
            for k, l in enumerate(range(1, 6, 1)):
                clf = DecisionTreeClassifier(max_depth=m, max_leaf_nodes=n, min_samples_leaf=l)
                t1 = time.time()
                ada_pre_pruning[i, j] = np.mean(cross_val_score(AdaBoostClassifier(clf), X, y))
                print(f'pruning time: adaBoost(max_depth={m}, max_leaf_nodes={n}) ' + str(time.time() - t1), file=f)
    print(f'pruning time: adaBoost ' + str(time.time() - t0), file=f)

    print(np.where(ada_pre_pruning == np.amax(ada_pre_pruning)), file=f)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf.fit(X_train, y_train)
    plt.figure()
    plot_tree(clf, filled=True)
    plt.savefig(folder + 'tree_ada_pre-prune_tree.png', format='png', bbox_inches="tight")
    plt.close()

    clf = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=100, min_samples_leaf=2)
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f'adaboost fitting time: ' + str(time.time() - t0), file=f)
    prune_index(clf.tree_, 0, 5)
    plot_learning_curve(AdaBoostClassifier(clf), X, y,'Ada Boosted Tree (after pruning)', 'tree_ada_after_sklearn_learning_curve.png', folder)

    plt.figure()
    plot_tree(clf, filled=True)
    plt.savefig(folder + 'tree_ada_post-prune_tree.png', format='png', bbox_inches="tight")
    plt.close()

    print('\nADA Learning Rates', file=f)
    learning_rates = [0.1, 0.5, 1, 1.5, 2, 4]
    for lr in learning_rates:
        t0 = time.time()
        plot_learning_curve(AdaBoostClassifier(clf, learning_rate=lr), X, y, f'Ada Boosted Tree (after pruning) LR={lr}', f'tree_ada_after_sklearn_learning_curve_{lr}.png', folder)
        print(f'5-fold CV for adaBoost: learning rate={lr} ' + str(time.time() - t0), file=f)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    plot_learning_curve(clf, X, y, 'Default Tree','tree_default_learning_curve.png', folder)

    clf_ada = AdaBoostClassifier(clf)
    clf_ada.fit(X_train, y_train)
    plot_learning_curve(clf_ada, X, y, 'ADA Boosted Tree', 'tree_ada_boosted_learning_curve.png',
                        folder)

# K Nearest Neighbor that tests multiple values of K and distance calculation
def knn(X, y, folder, f):
    print('\nK Nearest Neighbor', file=f)

    std = np.zeros(7, dtype=np.float16)
    mean = np.zeros(7, dtype=np.float16)

    for i, k in enumerate([1, 2, 3, 4, 5, 10, 20]):
        t0 = time.time()
        clf = KNeighborsClassifier(n_neighbors=k)
        plot_learning_curve(clf, X, y, f'KNN: K={k}', f'knn{k}_sklearn_learning_curve.png', folder)

        cv = cross_val_score(clf, X, y)

        std[i] = np.std(cv)
        m = np.mean(cv)
        mean[i] = m

        print(f'5-fold CV for KNN: k={k} ' + str(time.time() - t0), file=f)
        print(f'5-fold CV for KNN: k={k}, score={m}', file=f)

    plt.figure()
    plt.title('KNN Across Different Values for K')
    plt.xlabel('k')
    plt.ylabel('Accuracy')

    plt.fill_between([1, 2, 3, 4, 5, 10, 20], mean - std, mean + std, color='darkorange', alpha=0.1)
    plt.plot([1, 2, 3, 4, 5, 10, 20], mean, 'o-', color='darkorange')

    plt.savefig(folder + 'knn_k_vals.png', format='png')
    plt.close()


def svm(X, y, folder, f):
    print('SVM', file=f)

    i = [10, 20, 30, 40, 50, 100, 125, 250, 375, 500, 750, 1000]

    t0 = time.time()
    plot_learning_curve(SVC(kernel='linear', max_iter=-1), X, y,'SVM (linear kernel)', 'svm_linear_sklearn_learning_curve.png', folder)
    print(f'5-fold CV for SVM: Linear: ' + str(time.time() - t0), file=f)
    plot_iterations(SVC(kernel='linear'), X, y, i, 'SVM (linear kernel)', 'svm_linear_iter_learning_curve.png', folder)
    t0 = time.time()
    plot_learning_curve(SVC(kernel='rbf'), X, y,'SVM (rbf kernel)', 'svm_rbf_sklearn_learning_curve.png', folder)
    print(f'5-fold CV for SVM: rbf: ' + str(time.time() - t0), file=f)
    plot_iterations(SVC(kernel='rbf'), X, y, i, 'SVM (rbf kernel)', 'svm_rbf_iter_learning_curve.png', folder)
    t0 = time.time()
    plot_learning_curve(SVC(kernel='poly', degree=2), X, y,'SVM (polynomial kernel d2)', 'svm_poly2_sklearn_learning_curve.png', folder)
    print(f'5-fold CV for SVM: poly-2: ' + str(time.time() - t0), file=f)
    plot_iterations(SVC(kernel='poly', degree=2), X, y, i, 'SVM (polynomial kernel d2)',
                    'svm_poly2_iter_learning_cuWrve.png', folder)
    t0 = time.time()
    plot_learning_curve(SVC(kernel='poly', degree=5), X, y,'SVM (polynomial kernel d5)', 'svm_poly5_sklearn_learning_curve.png', folder)
    print(f'5-fold CV for SVM: poly-5: ' + str(time.time() - t0), file=f)
    plot_iterations(SVC(kernel='poly', degree=5), X, y, i, 'SVM (polynomial kernel d5)',
                    'svm_poly5_iter_learning_curve.png', folder)
    t0 = time.time()
    plot_learning_curve(SVC(kernel='poly', degree=12), X, y,'SVM (polynomial kernel d12)', 'svm_poly12_sklearn_learning_curve.png', folder)
    print(f'5-fold CV for SVM: poly-12: ' + str(time.time() - t0), file=f)
    plot_iterations(SVC(kernel='poly', degree=12), X, y, i, 'SVM (polynomial kernel d12)',
                    'svm_poly12_iter_learning_curve.png', folder)


def wine():
    folder = 'figures/wine/wine_'
    f = open("figures/wine_report.txt", "w")
    f.write("The Report for the ML Models for the Wine Dataset")
    f.write("")

    # load wine dataset
    print("Loading Wine Dataset")
    X, y = load_wine(return_X_y=True)
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print("Running DT")
    decision_tree(X, y, folder, f)

    print("Running NN")
    neural_networks(X_scaled, y, folder, f)

    print("Running Boosted DT")
    boosting_tree(X, y, folder, f)

    print("Running SVM")
    svm(X, y, folder, f)

    print("Running KNN")
    knn(X, y, folder, f)

    f.close()


def breast_cancer():
    folder = 'figures/breast_cancer/bc_'
    f = open("figures/breast_cancer_report.txt", "w")
    f.write("The Report for the ML Models for the Breast Cancer Dataset")
    f.write("")

    # load breast cancer dataset
    print("Loading Breast Cancer Dataset")
    X, y = load_breast_cancer(return_X_y=True)
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print("Running DT")
    decision_tree(X, y, folder, f)

    print("Running NN")
    neural_networks(X_scaled, y, folder, f)

    print("Running Boosted DT")
    boosting_tree(X, y, folder, f)

    print("Running SVM")
    svm(X, y, folder, f)

    print("Running KNN")
    knn(X, y, folder, f)

    f.close()


wine()
print()
breast_cancer()
