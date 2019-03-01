from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def model_report(clf, x_test, y_test):
    score = clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)
    paraResultItem = {}
    paraResultItem['score'] = score
    paraResultItem['precision'] = precision_score(y_test, y_pred)
    paraResultItem['recall'] = recall_score(y_test, y_pred)
    paraResultItem['f1'] = f1_score(y_test, y_pred)
    proba = clf.predict_proba(x_test)
    Y_prob = []
    for i in proba:
        Y_prob.append(i[1])
    paraResultItem['auc'] = roc_auc_score(y_test, Y_prob)
    return paraResultItem
