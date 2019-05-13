from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle
from data.prepare_data import prepare_data
import logging

logging.basicConfig(filename='/home/skye/Documents/Programming/sample-sklearn-ml-workflow.log',
                    filemode='w',
                    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')

def create_best_logreg(data_path, output_path):
    X, y, X_test, _ = prepare_data(data_path)
    X_train, X_valid, y_train, y_valid = train_test_split( X, y,
                                                           test_size=0.10,
                                                           random_state=42)
    pipe = Pipeline([
        ('feature_selection', SelectKBest(f_classif)),
        ('clf', LogisticRegression(random_state=2))])
    params = {'feature_selection__k': [12, 8, 4],
              'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(pipe, param_grid=params, scoring='roc_auc')
    #logging.INFO(str(grid_search.fit(X_train, y_train).best_params_))
    grid_search.fit(X_train, y_train)
    tn, fp, fn, tp = confusion_matrix(y_valid, grid_search.best_estimator_.predict(X_valid)).ravel()
    #logging.INFO("= Logistic Regression = ")
    #logging.INFO("Accuracy: {}".format(((tp + tn) / (tn + fp + fn + tp))))
    #logging.INFO("Recall: {}".format(((tp) / (fn + tp))))
    #logging.INFO("Specificity: {}".format(((tn) / (tn + fp))))
    #logging.INFO("F1 Score: {}".format(f1_score(y_valid, grid_search.best_estimator_.predict(X_valid))))
    #logging.INFO("AUC: {}".format(roc_auc_score(y_valid, grid_search.best_estimator_.predict(X_valid))))

    model_path = "{0}{1}".format(output_path, 'logreg.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file=file)

    return model_path, roc_auc_score(y_valid, grid_search.best_estimator_.predict(X_valid))