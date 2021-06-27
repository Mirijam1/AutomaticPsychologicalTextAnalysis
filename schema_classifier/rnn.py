import pickle

from numpy import interp

import utils
import sys

import numpy as np
import gensim
from gensim.models import KeyedVectors
import pandas as pd
import tensorflow as tf
import talos
from tensorflow import keras
from tensorflow.python.keras.metrics import Metric
import timeit
from keras.layers import Dense, Flatten, Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    classification_report, roc_curve, roc_auc_score, auc
from sklearn.model_selection import KFold

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(keras.__version__)
print(tf.__version__)

schemas = ["vulnerable", "angry", "impulsive", "happy", "detached", "punishing", "healthy"]
max_length = 500
max_words = 2000
vec_size = 300
t = Tokenizer(num_words=max_words)


# tokenizes and encodes the sentences.
def encode_and_pad(texts):
    print("ENCODING AND PADDING")
    t.fit_on_texts(texts)
    encoded_texts = t.texts_to_sequences(texts)
    return t, pad_sequences(encoded_texts, truncating="post", padding='post', maxlen=max_length)


# https://www.tensorflow.org/tutorials/text/word2vec#prepare_training_data_for_word2vec
def create_embedding_matrix(t):
    model = utils.get_word2vec()
    print('PREPARING EMBEDDING MATRIX')
    vocab_size = len(t.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, vec_size))
    for word, i in t.word_index.items():
        if model.__contains__(word):
            embedding_vector = model.__getitem__(word)
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return vocab_size, embedding_matrix


def fit_mlm_binary(train_X, train_y, val_X, val_y, params):
    # build the model
    model = Sequential()
    e = Embedding(vocab_size, vec_size, weights=[embedding_matrix], input_length=max_length, trainable=False)
    # embedding layer
    model.add(e)
    # LSTM layer
    model.add(Bidirectional(LSTM(params['lstm_units'])))
    # dropout layer
    model.add(Dropout(params['dropout']))
    # output layer
    model.add(Dense(7, activation='sigmoid'))
    # compile the model
    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['mean_absolute_error'])
    # summarize the model
    print(model.summary())
    # fit the model
    out = model.fit(train_X, train_y,
                    validation_data=[val_X, val_y],
                    batch_size=params['batch_size'],
                    epochs=100,
                    verbose=0)
    return out, model


# https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n
# https://stackoverflow.com/questions/59336899/which-loss-function-and-metrics-to-use-for-multi-label-classification-with-very
# https://github.com/keras-team/keras/issues/741
# https://github.com/keras-team/keras/issues/10371
def grid_search_mlm_binary(train_X, val_X, train_y, val_y):
    # define hyperparameter grid
    p = {'lstm_units': [100, 200, 300],
         'optimizer': ['rmsprop', 'Adam'],
         'dropout': [0.1, 0.2, 0.5],
         'batch_size': [32, 64]}
    # scan the grid
    tal = talos.Scan(x=train_X,
                     y=train_y,
                     x_val=val_X,
                     y_val=val_y,
                     model=fit_mlm_binary,
                     params=p,
                     experiment_name='fit_mlm_binary',
                     print_params=True,
                     clear_session=True)
    return tal


def fit_psm_binary(train_X, train_y, val_X, val_y, params):
    model = Sequential()
    e = Embedding(vocab_size, vec_size, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(LSTM(params['lstm_units'])))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['mean_absolute_error'])
    # summarize the model
    print(model.summary())
    # fit the model
    out = model.fit(train_X, train_y,
              validation_data=[val_X, val_y],
              batch_size=params['batch_size'],
              epochs=100,
              verbose=0,
              use_multiprocessing=True)
    return out, model


def grid_search_psm_binary(train_X, val_X, train_y, val_y):
    # define hyper-parameter grid
    p = {'lstm_units': [100, 200, 300],
         'optimizer': ['rmsprop', 'Adam'],
         'dropout': [0.1, 0.2, 0.5],
         'batch_size': [32, 64]}
    # scan the grid
    tal = talos.Scan(x=train_X,
                     y=train_y,
                     x_val=val_X,
                     y_val=val_y,
                     model=fit_psm_binary,
                     params=p,
                     experiment_name='fit_psm_binary',
                     print_params=True,
                     clear_session=True)
    return tal


def fit_psm_ordinal(train_X, train_y, test_X, test_y, params):
    # build the model
    model = Sequential()
    e = Embedding(vocab_size, vec_size, weights=[embedding_matrix], input_length=max_length, trainable=False)
    # embedding layer
    model.add(e)
    # LSTM layer
    model.add(Bidirectional(LSTM(params['lstm_units'])))
    # dropout layer
    model.add(Dropout(params['dropout']))
    # output layer
    model.add(Dense(4, activation='softmax'))
    # compile the model
    model.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['mean_absolute_error'])
    # summarize the model
    print(model.summary())
    # fit the model
    out = model.fit(train_X, train_y,
                    validation_data=[test_X, test_y],
                    batch_size=params['batch_size'],
                    epochs=100,
                    verbose=0)
    return out, model


def grid_search_psm_ordinal(train_X, test_X, train_y, test_y):
    # define hyperparameter grid
    p = {'lstm_units': [100, 200, 300],
         'optimizer': ['rmsprop', 'Adam'],
         'losses': ['categorical_crossentropy', 'mean_absolute_error'],
         'dropout': [0.1, 0.2, 0.5],
         'batch_size': [32, 64]}
    # scan the grid
    tal = talos.Scan(x=train_X,
                     y=train_y,
                     x_val=test_X,
                     y_val=test_y,
                     model=fit_psm_ordinal,
                     params=p,
                     experiment_name='fit_psm_ordinal',
                     print_params=True,
                     clear_session=True)
    return tal


# binary mlm model
def multilabel_model_binary(train_x, train_y, val_x, val_y):
    model = Sequential()
    e = Embedding(vocab_size, vec_size, weights=[embedding_matrix], input_length=max_length, trainable=False)
    # embedding layer
    model.add(e)
    # LSTM layer
    model.add(Bidirectional(LSTM(300)))
    # dropout layer to reduce overfitting
    model.add(Dropout(0.1))
    # output layer
    model.add(Dense(7, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mean_absolute_error'])
    out = model.fit(train_x, train_y,
                    validation_data=(val_x, val_y),
                    batch_size=32,
                    epochs=100,
                    verbose=1)
    loss, accuracy = model.evaluate(train_x, train_y, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))

    return out, model


# generate predictions with the multilabel_model
def predict_schema_mlm_binary(test_text, test_labels, fixed=None):
    if fixed is None:
        all_preds = np.zeros((test_text.shape[0], test_labels.shape[1], k))
        all_accurs = np.zeros((k, 7))
        for j in range(k):
            model_name = "../plots/rnn/MLMs/BINARY/mlm_" + str(j)
            model = keras.models.load_model(model_name + '.h5')
            preds = model.predict(test_text)
            preds = np.array([np.array([round(y) for y in x]) for x in preds])
            accurs = []
            for i in range(len(schemas)):
                accurs.append(accuracy_score(test_labels[:, i], preds[:, i]))
            all_preds[:, :, j] = preds
            all_accurs[j, :] = accurs
    else:
        model_name = "../plots/rnn/MLMs/BINARY/mlm_" + str(fixed)
        model = keras.models.load_model(model_name + '.h5')
        all_preds = model.predict(test_text)
        all_accurs = accuracy_score(test_labels, all_preds)
    return all_accurs, all_preds


# load single models
def load_single_models(directory):
    single_models = []
    for i in range(7):
        model_name = '/schema_model_' + schemas[i]
        get_from = directory + model_name
        model = keras.models.load_model(get_from + '.h5')
        single_models.append(model)
    return single_models


# binary psm model
def perschema_models_binary(train_x, train_y, val_x, val_y):
    model = Sequential()
    e = Embedding(vocab_size, vec_size, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(LSTM(200)))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mean_absolute_error'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(train_x, train_y,
              validation_data=[val_x, val_y],
              batch_size=32,
              epochs=100,
              verbose=0,
              use_multiprocessing=True)
    return model


# generate BINARY predictions with the per-schema models
def predict_schema_psm_binary(test_text, test_labels, fixed=None):
    if fixed is None:
        all_preds = np.zeros((test_labels.shape[0], test_labels.shape[1], k))
        all_accurs = np.zeros((k, 7))
        for j in range(k):
            directory_name = "../plots/rnn/PSMs/BINARY/psm_" + str(j)
            preds = np.zeros(test_labels.shape)
            accurs = []
            single_models = load_single_models(directory_name)
            for i in range(7):
                model = single_models[i]
                out = model.predict(test_text)
                out = [round(x[0]) for x in out]
                preds[:, i] = out
                # accuracy between predictions and test labels
                accurs.append(accuracy_score(test_labels[:, i], out))
            all_preds[:, :, j] = preds
            all_accurs[j, :] = accurs
    else:
        directory_name = "../plots/rnn/PSMs/BINARY/psm_" + str(fixed)
        all_preds = np.zeros(test_labels.shape)
        all_accurs = []
        single_models = load_single_models(directory_name)
        for i in range(7):
            model = single_models[i]
            out = model.predict(test_text)
            out = [round(x[0]) for x in out]
            all_preds[:, i] = out
            all_accurs.append(round(accuracy_score(test_labels[:, i], out)))
    return all_accurs, all_preds


# ordinal psm model
def perschema_models_ordinal(train_x, train_y, val_x, val_y):
    model = Sequential()
    e = Embedding(vocab_size, vec_size, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(LSTM(vec_size)))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))
    # compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mean_absolute_error'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(train_x, train_y,
              validation_data=[val_x, val_y],
              batch_size=32,
              epochs=10,
              verbose=0,
              use_multiprocessing=True)
    return model


# generate ORDINAL predictions with the per-schema models
def predict_schema_psm_ordinal(test_text, test_labels, fixed=None):
    if fixed is None:
        all_preds = np.zeros((test_labels.shape[0], test_labels.shape[1], k))
        all_gofs = np.zeros((k, 7))
        for j in range(k):
            directory_name = "../plots/rnn/PSMs/ORDINAL/psm_" + str(j)
            preds = np.zeros(test_labels.shape)
            gofs = []
            single_models = load_single_models(directory_name)
            for i in range(7):
                model = single_models[i]
                out = model.predict(test_text)
                out = [np.argmax(x) for x in out]
                preds[:, i] = out
                # spearman correlation between predictions and test labels
                gof, p = scipy.stats.spearmanr(out, test_labels[:, i])
                gofs.append(gof)
            all_preds[:, :, j] = preds
            all_gofs[j, :] = gofs
    else:
        directory_name = "../plots/rnn/PSMs/ORDINAL/psm_" + str(fixed)
        all_preds = np.zeros(test_labels.shape)
        all_gofs = []
        single_models = load_single_models(directory_name)
        for i in range(7):
            model = single_models[i]
            out = model.predict(test_text)
            out = [np.argmax(x) for x in out]
            all_preds[:, i] = out
            gof, p = scipy.stats.spearmanr(out, test_labels[:, i])
            all_gofs.append(gof)
    return all_gofs, all_preds


def kfold(k, train_x, train_y):
    acc_per_fold = []
    loss_per_fold = []

    # Define the K-fold Cross Validator
    kf = KFold(n_splits=k, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, val in kf.split(train_x, train_y):
        out, model = multilabel_model_binary(fold_no, train_x[train], train_y[train])
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        loss, accuracy = model.evaluate(train_x[val], train_y[val], verbose=0)
        print(f'Score for fold {fold_no}: Loss of {loss}; Accuracy of {accuracy * 100}%')
        acc_per_fold.append(accuracy * 100)
        loss_per_fold.append(loss)

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')
    return acc_per_fold, loss_per_fold


def multilabel_rnn_binary(padded_train_x, train_y, padded_test_x, test_y, padded_val_x, val_y):
    # train the multilabel model
    for i in range(k):
        # we train the model
        out, model = multilabel_model_binary(padded_train_x, train_y, padded_val_x, val_y)
        model.save("../plots/rnn/MLMs/BINARY/mlm_" + str(i) + '.h5')
        # Generate generalization metrics
        score = model.evaluate(padded_test_x, test_y, verbose=1)

        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        print(model.summary())

    accurs, preds = predict_schema_mlm_binary(padded_test_x, test_y)
    return accurs, preds


def perschema_rnn_binary(padded_train_x, train_y, padded_test_x, test_y, padded_val_x, val_y):
    # train the per schema models
    for j in range(k):
        directory_name = "../plots/rnn/PSMs/BINARY/psm_" + str(j)
        for i in range(7):
            train_label_schema = train_y[:, i]
            val_label_schema = val_y[:, i]
            model = perschema_models_binary(padded_train_x, train_label_schema, padded_val_x, val_label_schema)
            # we write trained models to files to free up working memory
            model_name = '/schema_model_' + schemas[i]
            save_model_under = directory_name + model_name
            model.save(save_model_under + '.h5')

    return predict_schema_psm_binary(padded_test_x, test_y)


def perschema_rnn_ordinal(padded_train_x, train_y, padded_test_x, test_y, padded_val_x, val_y):
    # train the per schema models
    for j in range(k):
        directory_name = "../plots/rnn/PSMs/ORDINAL/psm_" + str(j)
        for i in range(7):
            train_label_schema = np_utils.to_categorical(train_y[:, i], num_classes=4)
            val_label_schema = np_utils.to_categorical(val_y[:, i], num_classes=4)
            model = perschema_models_ordinal(padded_train_x, train_label_schema, padded_val_x, val_label_schema)
            # we write trained models to files to free up working memory
            model_name = '/schema_model_' + schemas[i]
            save_model_under = directory_name + model_name
            model.save(save_model_under + '.h5')

    accurs, preds = predict_schema_psm_ordinal(padded_test_x, test_y)
    return accurs, preds


def gof_spear(X, Y):
    # spearman correlation of columns (schemas)
    gof_spear = np.zeros(X.shape[1])
    for schema in range(7):
        res = scipy.stats.spearmanr(a=X[:, schema], b=Y[:, schema], nan_policy='raise')
        rho, p = scipy.stats.spearmanr(a=X[:, schema], b=Y[:, schema], nan_policy='raise')
        gof_spear[schema] = rho
    return gof_spear


def get_results_ordinal(gof, preds):
    # #make a sum of all classification values
    # gof_sum = np.sum(gof, axis=1)
    # #sort sums
    # gof_sum_sorted = np.sort(gof_sum)
    # #pick element that is closest but larger than median (we have even number of elements)
    # get_med_element = gof_sum_sorted[floor(k/2)]
    # #get index of median
    # gof_sum_med_idx = np.where(gof_sum == get_med_element)[0]
    # #choose this as the final model to use in H2 and to report in the paper
    # gof_out = gof[gof_sum_med_idx]
    # output_psm = np.transpose(gof_out)

    # print('RNN Multilabel Model Testset Output')
    # print(pd.DataFrame(data=gof, index=schemas, columns=['estimate']))

    print('Ordinal Per-Schema Model Testset Output')
    print(pd.DataFrame(data=gof[0], index=schemas, columns=['Spearman correlation']))


def get_results_binary(accurs, preds, test_y):
    print('Binary Model Testset Output')
    print(pd.DataFrame(data=accurs[0], index=schemas, columns=['estimate']))
    # Generate multiclass confusion matrices
    matrices = multilabel_confusion_matrix(test_y, preds)
    # Plotting matrices: code
    for i, mat in enumerate(matrices):
        cmd = ConfusionMatrixDisplay(mat, display_labels=np.unique(test_y)).plot()
        plt.title('Confusion Matrix for: ' + str(schemas[i]) + ", Accuracy: " + "{:.2%}"
                  .format(accuracy_score(y_true=test_y[:, i], y_pred=preds[:, i])))
        # plt.savefig("../plots/rnn/confusion_matrix_PSMs/confusion_matrix_" + schemas[i])
        plt.savefig("../plots/rnn/confusion_matrix_MLMs/confusion_matrix_" + schemas[i])
        plt.show()

    # Classification report
    report = classification_report(test_y, preds, target_names=schemas)
    print(report)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # ROC curve
    for i, schema in enumerate(schemas):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print('AUC: %.3f' % roc_auc[i])
        # # generate a no skill prediction (majority class)
        ns_preds = [0 for _ in range(len(test_y[:, i]))]
        # # # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(test_y[:, i], ns_preds)
        # plot the roc curve for each schema model
        plt.title('ROC Curve for: ' + schema + ", AUC: {:.3%}"
                  .format(roc_auc[i]))
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(fpr[i], tpr[i], marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # save the plot
        # plt.savefig("../plots/rnn/roc_PSMs/roc_" + schema)
        plt.savefig("../plots/rnn/roc_MLMs/roc_" + schema)
        # show the plot
        plt.show()

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(schemas))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(schemas)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(schemas)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # plot the roc curves for all models
    for j, schema in enumerate(schemas):
        plt.plot(fpr[j], tpr[j],
                 label='ROC curve: ' + schema + ', area = {1:0.2f}'
                 ''.format(j, roc_auc[j]))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # save the plot
    # plt.savefig("../plots/rnn/roc_PSMs/roc_all")
    plt.savefig("../plots/rnn/roc_MLMs/roc_all")
    # show the plot
    plt.show()

    tpr_file = open('../plots/rnn/tpr_rnn.pkl', 'wb')
    pickle.dump(tpr, tpr_file)

    fpr_file = open('../plots/rnn/fpr_rnn.pkl', 'wb')
    pickle.dump(fpr, fpr_file)

    roc_auc_file = open('../plots/rnn/roc_auc_rnn.pkl', 'wb')
    pickle.dump(roc_auc, roc_auc_file)



if __name__ == '__main__':
    df = pd.read_csv("../data/FINAL_CSV.csv")
    # BINARY LABELS
    # texts, labels_binary = utils.get_text_labels(df)
    # ORDINAL LABELS
    texts, labels_ordinal = utils.get_average_for_each_label(df)

    processed_texts, tokenized_texts = utils.pre_process_data(texts)
    print(processed_texts[0:5])
    print(tokenized_texts[0:5])
    t, padded_x = encode_and_pad(processed_texts)

    # binary
    # padded_train_x_binary, train_y_binary, padded_test_x_binary, test_y_binary, percent_train_test_binary = utils.split_data(
    #     padded_x, labels_binary, 0.2)
    # padded_train_x_binary, train_y_binary, padded_val_x_binary, val_y_binary, percent_train_val_binary = utils.split_data(
    #     padded_train_x_binary, train_y_binary, 0.1)
    # print("train binary: ", np.shape(padded_train_x_binary))
    # print("val binary: ", np.shape(padded_val_x_binary))
    # print("test binary: ", np.shape(padded_test_x_binary))
    # ordinal
    padded_train_x_ordinal, train_y_ordinal, padded_test_x_ordinal, test_y_ordinal, percent_train_test_ordinal = utils.split_data(
        padded_x, labels_ordinal, 0.2)
    padded_train_x_ordinal, train_y_ordinal, padded_val_x_ordinal, val_y_ordinal, percent_train_val_ordinal = utils.split_data(
        padded_train_x_ordinal, train_y_ordinal, 0.1)
    print("train ordinal: ", np.shape(padded_train_x_ordinal))
    print("val ordinal: ", np.shape(padded_val_x_ordinal))
    print("test ordinal: ", np.shape(padded_test_x_ordinal))
    #
    # create embedding matrix
    vocab_size, embedding_matrix = create_embedding_matrix(t)
    print("embedding matrix: ", np.shape(embedding_matrix))
    print("vocab_size: ", vocab_size)

    k = 1
    # acc_per_fold, loss_per_fold = kfold(k, padded_train_x, train_y)

    # Grid search
    start = timeit.timeit()
    for i, schema in enumerate(schemas):
        print(schema)
        sys.stdout = open('../plots/rnn/output_grid_search_psm_ordinal/' + schema + '.txt', 'x')
        train_label_schema = np_utils.to_categorical(train_y_ordinal[:, i], num_classes=4)
        val_label_schema = np_utils.to_categorical(val_y_ordinal[:, i], num_classes=4)
        # tal = grid_search_mlm_binary(padded_train_x, padded_val_x, train_y, val_y)
        tal = grid_search_psm_ordinal(padded_train_x_ordinal, padded_val_x_ordinal, train_label_schema, val_label_schema)
        # analyze the outcome
        analyze_object = talos.Analyze(tal)
        analysis_results = analyze_object.data
        # let's have a look at the results of the grid search
        print(analysis_results)
    sys.stdout = open('../plots/rnn/output_grid_search_psm_ordinal/elapsed_time.txt', 'x')
    end = timeit.timeit()
    print("time elapsed" + end - start)
    print(end - start)

    # print("binary multilabel_rnn()")
    # accurs, preds = multilabel_rnn_binary(padded_train_x_binary, train_y_binary, padded_test_x_binary, test_y_binary,
    #                       padded_val_x_binary, val_y_binary)
    # accurs, preds = predict_schema_mlm_binary(padded_test_x_binary, test_y_binary)
    # print("binary perschema_rnn()")
    # accurs, preds = perschema_rnn_binary(padded_train_x_binary, train_y_binary, padded_test_x_binary, test_y_binary
    #                                      , padded_val_x_binary, val_y_binary)
    # accurs, preds = predict_schema_psm_binary(padded_test_x_binary, test_y_binary)

    print("ordinal perschema_rnn()")
    accurs, preds = perschema_rnn_ordinal(padded_train_x_ordinal, train_y_ordinal, padded_test_x_ordinal, test_y_ordinal
                                          , padded_val_x_ordinal, val_y_ordinal)

    preds = preds.reshape(np.shape(preds)[0], 7)
    # get_results_binary(accurs, preds, test_y_binary)
    get_results_ordinal(accurs, preds)

    print("FINISH rnn.py")
