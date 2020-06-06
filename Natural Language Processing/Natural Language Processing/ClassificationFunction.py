# Testing Classification models


def naiveBayesClassification(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)
    
    
    
    # fitting classifier to training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    
    
    # predict test set result
    y_pred = classifier.predict(X_test)
    
    
    # to evaluate result
    # making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    print('Naive Bayes Classification: ')
    evalPerformance(TP, TN, FP, FN)

def LogisticClassification(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
    
    
    # fitting logistic regression to training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0, solver = 'lbfgs')
    classifier.fit(X_train, y_train)
    
    # predict test set result
    y_pred = classifier.predict(X_test)
    
    
    # to evaluate result
    # making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    print('Logistic Classification: ')
    evalPerformance(TP, TN, FP, FN)


def KNN_Classification(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    print('KNN Classification: ')
    evalPerformance(TP, TN, FP, FN)


def SvmClassification(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
    
    
    # Fitting classifier to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 'scale')
    classifier.fit(X_train, y_train)
    
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    print('SVM Classification: ')
    evalPerformance(TP, TN, FP, FN)
    
    

def KernelSvmClassification(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
    
    
    # fitting classifier to training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 'scale')
    classifier.fit(X_train, y_train)
    
    
    
    # predict test set result
    y_pred = classifier.predict(X_test)
    
    
    # to evaluate result
    # making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    print('NKernel SVM Classification: ')
    evalPerformance(TP, TN, FP, FN)
    

def DecisionTreeClassification(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
    
    
    # fitting classifier to training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    
    
    # predict test set result
    y_pred = classifier.predict(X_test)
    
    
    # to evaluate result
    # making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    print('Decision Tree Classification: ')
    evalPerformance(TP, TN, FP, FN)
    

def RandomForestClassification(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
    
    
    # fitting classifier to training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    
    
    # predict test set result
    y_pred = classifier.predict(X_test)
    
    
    # to evaluate result
    # making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    print('Random Forest Classification: ')
    evalPerformance(TP, TN, FP, FN)
    

def evalPerformance(TP, TN, FP, FN):
    print('Accuracy: ', (TP + TN) / (TP + TN + FP + FN))
    Precision = (TP / (TP + FP))
    Recall = (TP / (TP + FN))
    print('Precision: ', Precision)
    print('Recall: ', Recall)
    print('F1 Score: ', (2*Precision*Recall / (Precision + Recall)))
    print('*******************************************')