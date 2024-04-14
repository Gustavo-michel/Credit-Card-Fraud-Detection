import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, log_loss, make_scorer, f1_score, precision_score, recall_score


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

classifiers = [
    SVC(probability=True),
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
]

def preprocessor(X_train):
    numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
    )
    preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, X_train.columns)]
    )
    return preprocessor


def best_params_models():
    parameters = [{'kernel':('linear', 'rbf'), 'C':[1, 2, 10]}], # SVC
    [{}], # LogisticRegression
    [{}], # DecisionTreeClassifier
    [{}], # RandoForestClassifier
    [{}], # AdaBoostClassifier
    [{}], # LightGBM
    [{}], # SVM
    [{}], # Neural network
    scores = {'accuracy' :make_scorer(accuracy_score),
        'recall'   :make_scorer(recall_score),
        'precision':make_scorer(precision_score),
        'f1'       :make_scorer(f1_score)}
    best_params = []
    for classifier in classifiers:
       grid = GridSearchCV(estimator=classifier, param_grid=parameters[classifier], scoring=scores)
       best_params.append(grid.best_params_)

    print(pd.DataFrame(grid.cv_results_)[['params',
                                  'mean_test_recall',
                                  'mean_test_precision',
                                  'mean_test_f1']])
    return best_params


def validation_clf_models(X_train, X_test, y_train, y_test):
    for classifier in classifiers:
        pipe = Pipeline(steps=[("preprocessor", preprocessor(X_train=X_train)), ("classifier", classifier)])
        
        pipe.fit(X_train, y_train)
        predict_train = pipe.predict(X_train)
        predict_test = pipe.predict(X_test)

        print(classifier)
        print(' ')
        print(f"Roc Score train: {roc_auc_score(y_train, predict_train):.3f}")
        print(f"Log loss train: {log_loss(y_train, predict_train):.3f}")
        print(f"Roc Score train: {f1_score(y_train, predict_train):.3f}")
        print('- '*30)
        print(f"Roc Score test: {roc_auc_score(y_test, predict_test):.3f}")
        print(f"Log loss test: {log_loss(y_test, predict_test):.3f}")
        print(f"Roc Score test: {f1_score(y_test, predict_test):.3f}")
        print('-'*50)

def plot_validation_clf_models(X_train, X_test, y_train, y_test):
    for classifier in classifiers:
        pipe = Pipeline(steps=[("preprocessor", preprocessor(X_train=X_train)), ("classifier", classifier)])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)

        auc = round(roc_auc_score(y_test, y_pred), 4) 
        plt.plot(fpr, tpr, label=f'{classifier}, AUC={str(auc)}')
    plt.xlabel('Falso Positivo')
    plt.ylabel('verdadeiro Positivo')
    plt.title('Comparando modelos ROC-CURVE')
    plt.savefig("img/Comparando modelos ROC-CURVE")
    plt.legend(loc="lower right")
def evaluate():
    pass
       