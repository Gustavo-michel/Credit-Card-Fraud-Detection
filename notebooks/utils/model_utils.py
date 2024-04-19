import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, make_scorer, f1_score
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

classifiers = [
    SVC(probability=True),
    LogisticRegression(),
    RandomForestClassifier(),
    AdaBoostClassifier(algorithm='SAMME'),
]
def preprocessor(X):
    '''
     performs preprocessing
    '''
    numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
    )
    preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, X.columns)]
    )
    return preprocessor


def best_params_models():
    '''
        defines the best hyperparameters for the model using Grid search.
    '''
    parameters = [{'kernel':('linear', 'rbf'), 'C':[1, 2, 10]}], # SVC
    [{'penalty': ('l1','l2', 'elasticnet'), 'C':[1, 2, 10], 
      'solver':('bfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'saga')}], # LogisticRegression
    [{'n_estimators': [50, 100, 200], 'criterion': ('gini', 'entropy', 'log_loss')}], # RandomForestClassifier
    [{'n_estimators': [25, 50, 100,], 'learning_rate':  (0.1, 0.5, 1.0)}], # AdaBoostClassifier
    #[{}], # LightGBM

    scores = {'accuracy' :make_scorer(metrics.accuracy_score),
        'recall'   :make_scorer(metrics.recall_score),
        'precision':make_scorer(metrics.precision_score),
        'f1'       :make_scorer(f1_score)}
    
    print("0 = SVC, 1 = Logistic Regression, 2 = Random Forest, 4 = AdaBoost")
    select_clf = int(input("Selecione o Classificador: "))

    grid = GridSearchCV(estimator=classifiers[select_clf], param_grid=parameters[select_clf], scoring=scores)
    best_params = []
    best_params.append(grid.best_params_)

    print(pd.DataFrame(grid.cv_results_)[['params',
                                  'mean_test_recall',
                                  'mean_test_precision',
                                  'mean_test_f1']])
    return best_params


def validation_clf_models(X_train, X_test, y_train, y_test):
    '''
     performs model validation with scores.
    '''
    for classifier in classifiers:
        pipe = Pipeline(steps=[("preprocessor", preprocessor(X=X_train)), ("classifier", classifier)])
        
        pipe.fit(X_train, y_train)
        predict_train = pipe.predict(X_train)
        predict_test = pipe.predict(X_test)

        print(classifier)
        print(' ')
        print(f"F1 Score train: {f1_score(y_train, predict_train):.3f}")
        print(f"Accuracy train: {metrics.accuracy_score(y_train, predict_train):.3f}")
        print(f"Recall train: {metrics.recall_score(y_train, predict_train):.3f}")
        print(f"Precision train: {metrics.precision_score(y_train, predict_train):.3f}")
        print(f"Log loss train: {log_loss(y_train, predict_train):.3f}")
        print('- '*30)
        print(f"F1 Score test: {f1_score(y_test, predict_test):.3f}")
        print(f"Accuracy test: {metrics.accuracy_score(y_test, predict_test):.3f}")
        print(f"Recall test: {metrics.recall_score(y_test, predict_test):.3f}")
        print(f"Precision test: {metrics.precision_score(y_test, predict_test):.3f}")
        print(f"Log loss test: {log_loss(y_test, predict_test):.3f}")
        print('-'*50)

def plot_validation_clf_models(X_train, X_test, y_train, y_test):
    '''
     Plot Score of models with ROC Curve
    '''
    for classifier in classifiers:
        pipe = Pipeline(steps=[("preprocessor", preprocessor(X=X_train)), ("classifier", classifier)])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)

        auc = round(roc_auc_score(y_test, y_pred), 4) 
        plt.plot(fpr, tpr, label=f'{classifier}, AUC={str(auc)}')
    plt.xlabel('Falso Positivo')
    plt.ylabel('verdadeiro Positivo')
    plt.title('Comparando modelos ROC-CURVE')
    plt.legend(loc="lower right")
    plt.savefig("img/Comparando modelos ROC-CURVE")


def evaluate():
    # print(confusion_matrix(Y_test, predictions))
    # print(classification_report(Y_test, predictions))
    pass
       