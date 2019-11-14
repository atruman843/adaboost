from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import csv
import numpy

data = []
labels = []

def boost():

    ###########################################################################
    ## This piece of code will allow you to visualize the margin in a        ##
    ## scatter plot.                                                         ##
    ###########################################################################

    clf = AdaBoostClassifier(
        algorithm='SAMME',
        n_estimators=10
    )

    clf.fit(data, labels)
    alphas1 = clf.estimator_weights_
    alphas1 = alphas1/min(alphas1)
    alphas1 = alphas1/sum(alphas1)
    sums = [0] * 345
    for ind1, estimator in enumerate(clf.estimators_):
        for ind2, prediction in enumerate(estimator.predict(data)):
            sums[ind2] += (float(prediction) * float(alphas1[ind1]))
    x1 = numpy.sort(sums)
    y1 = numpy.arange(1, (len(x1)+1))
    pyplot.plot(x1, y1, c='b', marker='s', linestyle='none', label='10')

    clf = AdaBoostClassifier(
        algorithm='SAMME',
        n_estimators=50
    )

    clf.fit(data, labels)
    alphas2 = clf.estimator_weights_
    alphas2 = alphas2/min(alphas2)
    alphas2 = alphas2/sum(alphas2)
    sums = [0] * 345
    for ind1, estimator in enumerate(clf.estimators_):
        for ind2, prediction in enumerate(estimator.predict(data)):
            sums[ind2] += (float(prediction) * float(alphas2[ind1]))
    x2 = numpy.sort(sums)
    y2 = numpy.arange(1, (len(x2)+1))
    pyplot.plot(x2, y2, c='r', marker='o', linestyle='none', label='50')

    clf = AdaBoostClassifier(
        algorithm='SAMME',
        n_estimators=100
    )

    clf.fit(data, labels)
    alphas3 = clf.estimator_weights_
    alphas3 = alphas3/min(alphas3)
    alphas3 = alphas3/sum(alphas3)
    sums = [0] * 345
    for ind1, estimator in enumerate(clf.estimators_):
        for ind2, prediction in enumerate(estimator.predict(data)):
            sums[ind2] += (float(prediction) * float(alphas3[ind1]))
    x3 = numpy.sort(sums)
    y3 = numpy.arange(1, (len(x3)+1))
    pyplot.plot(x3, y3, c='g', marker='d', linestyle='none', label='100')

    pyplot.gca().legend(('10', '50', '100'))
    pyplot.show()

    ###########################################################################
    ## This piece of code will allow you to view the feature used for        ##
    ## each decision tree, the avaialble classes, and the threshold value    ##
    ###########################################################################

    # clf = AdaBoostClassifier(
    #     algorithm='SAMME',
    #     n_estimators=100
    # )
    #
    # clf.fit(data, labels)
    # for i, estimator in enumerate(clf.estimators_):
    #     print "{feature} {classes}".format(feature=estimator.feature_importances_, classes=estimator.classes_)
    #     if i == 10:
    #         break;
    #
    # for i, decision in enumerate(clf.staged_decision_function(data)):
    #     print "{threshold}".format(threshold=decision[-1])
    #     if i == 10:
    #         break;

    ###########################################################################
    ## This piece of code will allow you to view the training and test       ##
    ## error for each boosting iteration.                                    ##
    ###########################################################################

    # test_scores = [0] * 100
    # train_scores = [0] * 100
    # clf = AdaBoostClassifier(
    #     algorithm='SAMME',
    #     n_estimators=100
    # )
    #
    # for i in range(0, 50):
    #     x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.9)
    #     clf.fit(x_train, y_train)
    #     for ind, score in enumerate(clf.staged_score(x_train, y_train)):
    #         train_scores[ind] += score
    #     for ind, score in enumerate(clf.staged_score(x_test, y_test)):
    #         test_scores[ind] += score
    #
    # for i in range(0, 100):
    #     train_scores[i] /= 50
    #     test_scores[i]  /= 50
    #     print "{train} {test}".format(train=(1-train_scores[i]), test=(1-test_scores[i]))

def read_file():
    file = open("data/bupa.csv", "r")
    for line in csv.reader(file, delimiter=','):
        data.append(line[:6])
        labels.append(line[-1])

read_file()
boost()
