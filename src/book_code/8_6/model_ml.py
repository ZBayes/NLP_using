from sklearn.svm import SVC

def model_ml(model_num, x_train, y_train):
	clf = ""
	model_msg = ""

	if model_num == "1":
		model_msg = "model1: SVM, kernel=rbf, shrinking=True"
		clf = SVC(kernel='rbf', probability=True,
		          shrinking=True)
		clf.fit(x_train, y_train)
	if model_msg != "":
		return clf, model_msg