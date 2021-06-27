import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
from autocorrect import Speller
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import scipy.stats

import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import schema_classifier.myutils as utils

spell = Speller(lang='en')
nltk.download('stopwords')
nltk.download('punkt')
tk = RegexpTokenizer(r'\w+')


schemas = ["vulnerable", "angry", "impulsive", "happy", "detached", "punishing", "healthy"]
num_of_schemas = 7
max_words = 2000
max_epochs = 30
vec_size = 500

def svm_fast_text():
    df = pd.read_csv("../data/FINAL_CSV.csv")
    texts, labels = utils.get_text_labels(df)
    processed, tokenized = utils.pre_process_data(texts)
    fastText_model = utils.training_model_fast_text()

    vectors = []
    for row in range(len(tokenized)):
        words = texts[row]
        row_vec = np.zeros(fastText_model.get_dimension())
        for each in words:
            vector = fastText_model[each]
            row_vec = np.add(row_vec, vector)
        if len(words) > 0:
            row_vec /= len(words)
            vectors.append(row_vec)
        else:
            # print("why length = 0", row_vec, words)
            vectors.append(row_vec)

    X = vectors
    np_x = np.asarray(X)
    np_y = np.asarray(labels)

    x_train, y_train, x_test, y_test, test_percent = utils.split_data(np_x, np_y, 0.2)
    scaled_train = svm_scaler(x_train)

    # svm
    svm_rbf_models = svm_classification(x_train, y_train, scaled_train, 'rbf')
    print("RBF")
    svm_rbf_out = svm_predict(svm_rbf_models, x_test, y_train, y_test, scaled_train, 'rbf')
    svm_lin_models = svm_classification(x_train, y_train, scaled_train, 'linear')
    print("Linear")
    svm_lin_out = svm_predict(svm_lin_models, x_test, y_train, y_test, scaled_train, 'linear')
    svm_poly_models = svm_classification(x_train, y_train, scaled_train, 'poly')
    print("Poly")
    svm_poly_out = svm_predict(svm_poly_models, x_test, y_train, y_test, scaled_train, 'poly')


def gof_spear(X, Y):
    # spearman correlation of columns (schemas)
    gof_spear = np.zeros(X.shape[1])
    for schema in range(num_of_schemas):
        rho, p = spearmanr(X[:, schema], Y[:, schema])
        gof_spear[schema] = rho
    return gof_spear

# weighting model output (spearman correlations) by schema frequencies in training set and returning mean over schemas
def performance(train_y,output):
    train_y = np.array(train_y)
    train_y[train_y>0]=1
    weighting = train_y.sum(axis=0)/train_y.shape[0]
    perf = output * weighting
    return np.nanmean(np.array(perf), axis=0)

def svm_scaler(train_X):
        # scale the data
        scaler_texts = StandardScaler()
        scaler_texts = scaler_texts.fit(train_X)
        return scaler_texts

def svm_classification(train_X, train_y, text_scaler, kern):
    models = []
    train_X = text_scaler.transform(train_X)
    # fit a new support vector regression for each schema
    for schema in range(len(schemas)):
        model = svm.SVC(kernel=kern)
        model.fit(train_X, train_y[:, schema])
        models.append(model)
    return models

def svm_predict(svm_models,test_X,train_y,test_y,text_scaler, kern_name):
    #empty array to collect the results (should have shape of samples to classify)
    votes = np.zeros(test_y.shape)
    for schema in range(num_of_schemas):
        svm_model=svm_models[schema]
        prediction = svm_model.predict(text_scaler.transform(test_X))
        votes[:,schema] = prediction

    out_arr = []
    for schema in range(num_of_schemas):
        out = accuracy_score(votes[:, schema], test_y[:, schema])
        print(schemas[schema], out)
        out_arr.append(out)

    plot_data(out_arr, kern_name)

    return out

def plot_data(result, kernel_name):
    plt.xlabel('Schemas')
    plt.ylabel('Accuracy')
    plt.bar(schemas, result, color=['red', 'blue', 'purple', 'green', 'orange', 'lavender', 'yellow'])
    if kernel_name is 'rbf':
        plt.title('RBF')
    elif kernel_name is 'poly':
        plt.title('Polynomial')
    else:
        plt.title('Linear')
    plt.savefig('../plots/svm/{}'.format(kernel_name))
    plt.show()

# From Jah
def svm_diff_approach():
    randomSeed = 42
    df = pd.read_csv("../data/FINAL_CSV.csv")
    texts, labels = utils.get_text_labels(df)
    processed, tokenized = utils.pre_process_data(texts)
    fastText_model = utils.training_model_fast_text()

    vectors = []
    for row in range(len(tokenized)):
        words = texts[row]
        row_vec = np.zeros(fastText_model.get_dimension())
        for each in words:
            vector = fastText_model[each]
            row_vec = np.add(row_vec, vector)
        if len(words) > 0:
            row_vec /= len(words)
            vectors.append(row_vec)
        else:
            # print("why length = 0", row_vec, words)
            vectors.append(row_vec)

    X = vectors
    np_x = np.asarray(X)
    np_y = np.asarray(labels)

    x_train, y_train, x_test, y_test, test_percent = utils.split_data(np_x, np_y, 0.2)
    # scaled_train = svm_scaler(x_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # x_train = scaled_train.transform(x_train)
    # x_test = scaled_train.transform(x_test)
    # Create the SVM
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}


    svm_model = svm.SVC(random_state=randomSeed, C=1000, gamma=5, kernel='rbf')
    # clf = CalibratedClassifierCV(svm_model)

    # Make it an Multilabel classifier
    multilabel_classifier = MultiOutputClassifier(svm_model, n_jobs=-1)

    # Fit the data to the Multilabel classifier
    multilabel_classifier = multilabel_classifier.fit(x_train, y_train)

    # Get predictions for test data
    schema_test_pred = multilabel_classifier.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, schema_test_pred))
    print('F1 Score: ', f1_score(y_test, schema_test_pred, average='weighted'))
    print(classification_report(y_test, schema_test_pred, target_names=schemas))

    # Generate multiclass confusion matrices
    matrices = multilabel_confusion_matrix(y_test, schema_test_pred)
    # Plotting matrices: code
    i = 1
    for mat in matrices:
        cmd = ConfusionMatrixDisplay(mat, display_labels=np.unique(y_test)).plot()
        plt.title('Confusion Matrix for: ' + str(schemas[i-1]) + ", Accuracy: " + "{:.2%}".format(accuracy_score(y_true=y_test[:,i-1], y_pred=schema_test_pred[:,i-1])))
        plt.show()
        i += 1


def svm_doc2vec():
    df = pd.read_csv("../data/FINAL_CSV.csv")
    texts, labels = utils.get_text_labels(df)
    processed, tokenized = utils.pre_process_data(texts)
    doc2vec_model = utils.training_model_d2v()

    vectors = []
    for row in range(len(tokenized)):
        words = texts[row]
        row_vec = np.zeros(doc2vec_model.get_dimension())
        for each in words:
            vector = doc2vec_model[each]
            row_vec = np.add(row_vec, vector)
        if len(words) > 0:
            row_vec /= len(words)
            vectors.append(row_vec)
        else:
            # print("why length = 0", row_vec, words)
            vectors.append(row_vec)

    X = vectors
    np_x = np.asarray(X)
    np_y = np.asarray(labels)

    x_train, y_train, x_test, y_test, test_percent = utils.split_data(np_x, np_y, 0.2)
    scaled_train = svm_scaler(x_train)

    # svm
    svm_rbf_models = svm_classification(x_train, y_train, scaled_train, 'rbf')
    print("RBF")
    svm_rbf_out = svm_predict(svm_rbf_models, x_test, y_train, y_test, scaled_train, 'rbf')
    svm_lin_models = svm_classification(x_train, y_train, scaled_train, 'linear')
    print("Linear")
    svm_lin_out = svm_predict(svm_lin_models, x_test, y_train, y_test, scaled_train, 'linear')
    svm_poly_models = svm_classification(x_train, y_train, scaled_train, 'poly')
    print("Poly")
    svm_poly_out = svm_predict(svm_poly_models, x_test, y_train, y_test, scaled_train, 'poly')

#### Goodness of Fit
def gof_spear(X,Y):
    #spearman correlation of columns (schemas)
    gof_spear = np.zeros(X.shape[1])
    for schema in range(9):
        rho,p = scipy.stats.spearmanr(X[:,schema],Y[:,schema])
        gof_spear[schema]=rho
    return gof_spear

svm_diff_approach()
