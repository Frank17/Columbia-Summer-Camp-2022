from typing import Optional, Tuple, Callable
from operator import itemgetter
from time import perf_counter

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Feature engineering
from sklearn.model_selection import train_test_split

# ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Cleaned dataset
from dataset import all_features, target_features

# formatting constants
DIVIDER = '——' * 60
COLOR_DICT = {
    'default':          '\033[0m',          'black':            '\033[0;30m',
    'red':              '\033[0;31m',       'green':            '\033[0;32m',
    'yellow':           '\033[0;33m',       'blue':             '\033[0;34m'
}

# dicts that contain the accuracy score and time spend of every algo
_acc_scores = {}
_time_spent = {}

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    all_features, target_features, test_size=0.25, random_state=42
)


def turncolor(color: str) -> None:
    """
    Make the output text colorful
    """
    print(COLOR_DICT.get(color, ''), end='')


def bold(text: str) -> str:
    return f'\033[1m{text}\033[0m'


def italicize(text: str) -> str:
    return f'\033[3m{text}\033[0m'


def train_logrog() -> Tuple[float, float]:
    """
    Training with the logistic regression model
    """
    logreg_classifier = LogisticRegression()
    logreg_classifier.fit(X_train, y_train)
    y_pred = logreg_classifier.predict(X_test)
    score = logreg_classifier.score(X_test, y_test)
    return y_pred, score


def train_sgd() -> Tuple[float, float]:
    """
    Training with the stochastic gradient descent model
    """
    sgd_classifier = SGDClassifier(
        loss='hinge',
        penalty='l2',
        max_iter=120
    )
    sgd_classifier.fit(X_train, y_train)
    y_pred = sgd_classifier.predict(X_test)
    score = sgd_classifier.score(X_test, y_test)
    return y_pred, score


def train_svm() -> Tuple[float, float]:
    """
    Training with the support vector machine model
    """
    svm_classifier = SVC(C=0.5)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    score = svm_classifier.score(X_test, y_test)
    return y_pred, score


def train_dt() -> Tuple[float, float]:
    """
    Training with the decision tree model
    """
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    score = dt_classifier.score(X_test, y_test)
    return y_pred, score


def train_rf() -> Tuple[float, float]:
    """
    Training with the random forest model
    """
    rf_classifier = RandomForestClassifier(
        criterion='gini',
        n_estimators=600,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='auto',
        oob_score=True,
        n_jobs=-1
    )
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    score = rf_classifier.score(X_test, y_test)
    return y_pred, score


def train_xgb() -> Tuple[float, float]:
    """
    Training with the eXtreme Gradient Boosting model
        - Hypertuning param: learning_rate
    """
    def remove_invalid_chars(colname):
        return colname.replace('[', '').replace(']', '').replace('<', '')

    clean_X_train = X_train.rename(remove_invalid_chars, axis=1)
    clean_X_test = X_test.rename(remove_invalid_chars, axis=1)

    xgb_classifier = XGBClassifier(
        n_estimators=400,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    xgb_classifier.fit(clean_X_train, y_train)
    y_pred = xgb_classifier.predict(clean_X_test)
    score = xgb_classifier.score(clean_X_test, y_test)
    return y_pred, score


def train_gb() -> Tuple[float, float]:
    """
    Training with the gradient boosting model
        - Hypotuning: learning_rate
    """
    gb_classifier = GradientBoostingClassifier(
        n_estimators=400
    )
    gb_classifier.fit(X_train, y_train)
    y_pred = gb_classifier.predict(X_test)
    score = gb_classifier.score(X_test, y_test)
    return y_pred, score


def train_knn() -> Tuple[float, float]:
    """
    Training with the K-nearest neighbors model
        - Hypotuning: n_neighbors
    """
    knn_classifer = KNeighborsClassifier()
    knn_classifer.fit(X_train, y_train)
    y_pred = knn_classifer.predict(X_test)
    score = knn_classifer.score(X_test, y_test)
    return y_pred, score



def train_nb() -> Tuple[float, float]:
    """
    Training with the naive bayes model
    """
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    score = nb_classifier.score(X_test, y_test)
    return y_pred, score
    

def evaluate(
    model_name: str,
    y_pred: float,
    estimator_score: Optional[int] = None,
    hide_output: bool = False,
) -> None:
    """
    Evaluate the accuracy of the model via accuracy score, text report and
                                           confusion matrix
    """
    acc_score = estimator_score or accuracy_score((y_pred, y_test) * 100, 2)
    cf_mtx = confusion_matrix(y_test, y_pred)

    _acc_scores[model_name] = acc_score           # update the global acc_scores list

    if not hide_output:
        turncolor('red')
        print(bold('<-- Model Accuracy -->'))
        print(f'The accuracy score of the {model_name} is: {acc_score:.3f}', '\n')

        turncolor('blue')
        print(bold('<-- Classification Report -->'))
        print(classification_report(y_pred, y_test, digits=3), '\n')

        turncolor('green')
        print(bold('<-- Heat Map -->'))
        sns.heatmap(cf_mtx.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.show()

        turncolor('black')
        print(bold(DIVIDER))
        turncolor('default')


def test_all(hide_output: bool = False) -> None:
    """
    Test every model implemented, print out their output (only if hide_output
    is set to False, otherwise don't display anything), and store some data
    to the global dicts (_acc_scores and _time_spent)
    """
    models = {
        # 'Logistic Regression': train_logrog,
        'Stochastic Gradient Descent': train_sgd,
        # 'Support Vector Machine': train_svm,
        # 'Decision Tree': train_dt,
        # 'Random Forest': train_rf,
        # 'XGBoost': train_xgb,
        # 'Gradient Boosting': train_gb,
        # 'K-Nearest Neighbors': train_knn,
        # 'Naive Bayes': train_nb
    }
    for name, train_func in models.items():
        start = perf_counter()
        y_outputs = train_func()
        end = perf_counter()
        _time_spent[name] = (end - start)
        evaluate(name, *y_outputs, hide_output=hide_output)


def conclude() -> None:
    """
    Print out the most/least accurate algorithms, as well as the most/least
    time-consuming algorithms
    """
    def get_extreme(f: Callable, d: dict) -> float:
        return f(d.items(), key=itemgetter(1))

    most_acc_name, most_acc_score = get_extreme(max, _acc_scores)
    least_acc_name, least_acc_score = get_extreme(min, _acc_scores)

    most_time_name, most_time_score = get_extreme(max, _time_spent)
    least_time_name, least_time_score = get_extreme(min, _time_spent)
    
    print(
        italicize(
            f'- The most accurate algorithm is        {most_acc_name:^30}'
            f'with the score of  {most_acc_score:.1%}\n'
            f'- The least accurate algorithm is       {least_acc_name:^30}'
            f'with the score of  {least_acc_score:.1%}\n'
            '\n'
            f'- The most time-consuming algorithm is  {most_time_name:^30}'
            f'with the length of {most_time_score:.3f} seconds\n'
            f'- The least time-consuming algorithm is {least_time_name:^30}'
            f'with the length of {least_time_score:.3f} seconds\n'
        ) +
        bold(DIVIDER)
    )


def summarize() -> None:
    sorted_scores = sorted(
        _acc_scores.items(), key=itemgetter(1), reverse=True
    )
    for name, acc_score in sorted_scores:
        time_spent = _time_spent[name]
        print(
            f'👉 {name:^30} achieves an accuracy score of {acc_score:^3.1%}'
            f' and takes {time_spent:>6.3f} seconds to train.'
        )


test_all(hide_output=True)
conclude()
summarize()
