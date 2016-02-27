
def classify(features, labels):
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()

    clf.fit(features, labels)

    return clf




def accuracy(features_train, labels_train, features_test, labels_test):

    from time import time

    t0=time()
    clf = classify(features_train, labels_train)
    print "classification(training) time:", round(time()-t0, 3), "s"

    t0=time()
    pred = clf.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"

    t0=time()
    accuracy = clf.score(features_test, labels_test)
    print "scoring time:", round(time()-t0, 3), "s"

    return accuracy

