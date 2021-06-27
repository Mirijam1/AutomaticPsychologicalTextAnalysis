import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import gensim
import contractions
import math
import numpy as np
import sklearn
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import utils

import matplotlib.pyplot as plt

# nltk.download('stopwords')

schemas = ["vulnerable", "angry", "impulsive", "happy", "detached", "punishing", "healthy"]
colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]
num_of_schemas = 7
max_words = 2000
max_epochs = 30
vec_size = 500
tk = RegexpTokenizer(r'\w+')


def training_model_d2v(texts):
    tagged_docs = read_corpus(texts)
    print("TRAINING MODEL")

    model = gensim.models.Doc2Vec(documents=tagged_docs, vector_size=vec_size, window=10, epochs=max_epochs, min_count=1,
                                  workers=4, alpha=0.025, min_alpha=0.025)
    model.save('../models/schema-d2v-knn.model')


def read_corpus(texts):
    i = 0
    result = []
    # associate tag to each document
    for sentence in texts:
        result.append(gensim.models.doc2vec.TaggedDocument(sentence, [i]))
        i += 1
    return result


def remove_stopwords(s: str) -> str:
    new_str = ""
    for word in s.split():
        if word not in stopwords.words('english'):
            new_str += word + " "
    return new_str


def my_tokenize(texts: list) -> list:
    tokenized_texts = []
    for i in range(len(texts)):
        words = tk.tokenize(texts[i])
        tokenized_texts.append(words)
    return tokenized_texts


def pre_process_data(texts: list) -> list:
    # Convert all to lowercase
    processed_texts = list(map(lambda s: s.lower(), texts))

    # TODO: Noise removal
    processed_texts = list(map(lambda s: contractions.fix(s), processed_texts))

    # TODO: Spelling correction

    # TODO: Stop word removal
    processed_texts = list(map(lambda s: remove_stopwords(s), processed_texts))

    return processed_texts


def my_mlknn(model: gensim.models.doc2vec.Doc2Vec, labels):
    # X = model.docvecs.doctag_syn0
    X = model.docvecs.vectors_docs
    np_x = np.asarray(X)
    np_y = np.asarray(labels)

    s = np.arange(np_x.shape[0])
    np.random.shuffle(s)
    coordinates = np_x[s]
    tags = np_y[s]

    train = math.floor(len(X) * 0.75)
    print("train index: " + str(train))
    x_train = np_x[:train]
    x_test = np_x[train:]
    y_train = np_y[:train]
    y_test = np_y[train:]

    mlknn = MLkNN(k=40)

    mlknn.fit(x_train, y_train)
    print("FITTED")

    y_pred = mlknn.predict(x_test)
    print("PREDICTION")
    print("K=" + str(mlknn.k) + ", accuracy score: " + str(accuracy_score(y_test, y_pred)))


def my_knn(model: gensim.models.doc2vec.Doc2Vec, labels):
    for i, schema in enumerate(schemas):
        X = model.docvecs.vectors_docs
        np_x = np.asarray(X)
        np_y = np.asarray(labels[:, i])

        # shuffle
        s = np.arange(np_x.shape[0])
        np.random.shuffle(s)
        np_x_s = np_x[s]
        np_y_s = np_y[s]

        train = math.floor(len(X) * 0.75)

        x_train = np_x_s[:train]
        x_test = np_x_s[train:]
        y_train = np_y_s[:train]
        y_test = np_y_s[train:]

        knn = KNeighborsClassifier(n_neighbors=40)

        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        print(schema + ": " + str(accuracy_score(y_test, y_pred)))


def knn_error_plot(x_train, y_train, x_test, y_test):
    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 51):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 51), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=5)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()


def k_means_scatter_plot(model):
    # create and apply PCA transform
    X = model.docvecs.vectors_docs
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=7, max_iter=100)
    kmeans.fit(principal_components)
    y_kmeans = kmeans.predict(principal_components)
    # plot data with seaborn
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y_kmeans, s=15, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    plt.title("Doc2Vec vectorspace (Unlabelled samples + PCA)")
    plt.show()


def get_knn_accuracy(model, labels):
    accuracy = []

    for i, schema in enumerate(schemas):
        X = model.docvecs.vectors_docs
        np_x = np.asarray(X)
        np_y = np.asarray(labels[:, i])

        # shuffle
        s = np.arange(np_x.shape[0])
        np.random.shuffle(s)
        np_x_s = np_x[s]
        np_y_s = np_y[s]

        train = math.floor(len(X) * 0.85)

        x_train = np_x_s[:train]
        x_test = np_x_s[train:]
        y_train = np_y_s[:train]
        y_test = np_y_s[train:]

        knn = KNeighborsClassifier(n_neighbors=40)

        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        accuracy.append(round(accuracy_score(y_test, y_pred), 2))
    return accuracy


def print_accuracy(accuracy):
    for i, schema in enumerate(schemas):
        print(schema + ": " + str(accuracy[i]))

    avg_accuracy = 0
    for acc in accuracy:
        avg_accuracy += acc

    avg_accuracy = avg_accuracy / len(accuracy)
    print("Mean_accuracy" + ": " + str(round(avg_accuracy, 2)))


def knn_accuracy_plot(model, labels):
    number_of_k = 70
    iterations = 10
    # avg_accuracy = [0] * number_of_k

    plt.figure(figsize=(12, 6))

    for i, schema in enumerate(schemas):
        accuracy = [0] * number_of_k

        # 10 iterations
        for j in range(0, iterations):
            X = model.docvecs.vectors_docs

            np_x = np.asarray(X)
            np_y = np.asarray(labels[:, i])

            # shuffle
            s = np.arange(np_x.shape[0])
            np.random.shuffle(s)
            np_x_s = np_x[s]
            np_y_s = np_y[s]

            train = math.floor(len(X) * 0.75)

            x_train = np_x_s[:train]
            x_test = np_x_s[train:]
            y_train = np_y_s[:train]
            y_test = np_y_s[train:]

            # Calculating error for K values between 1 and 80
            for k in range(1, number_of_k + 1):
                mlknn = MLkNN(k=k)
                mlknn.fit(x_train, y_train)
                y_pred = mlknn.predict(x_test)
                accuracy[k - 1] += (mlknn.score(y_pred, y_test))

        for j in range(0, number_of_k):
            accuracy[j] /= iterations
            # avg_accuracy[j] += accuracy[j]

        plt.plot(range(1, number_of_k + 1), accuracy, color=colors[i], linestyle='solid', label=schemas[i])

def mlknn_accuracy_plot(model, labels):
    number_of_k = 70
    iterations = 10
    # avg_accuracy = [0] * number_of_k

    plt.figure(figsize=(12, 6))


    accuracy = [0] * number_of_k

    # 10 iterations
    for j in range(0, iterations):
        X = model.docvecs.vectors_docs

        np_x = np.asarray(X)
        np_y = np.asarray(labels)

        # shuffle
        s = np.arange(np_x.shape[0])
        np.random.shuffle(s)
        np_x_s = np_x[s]
        np_y_s = np_y[s]

        train = math.floor(len(X) * 0.85)

        x_train = np_x_s[:train]
        x_test = np_x_s[train:]
        y_train = np_y_s[:train]
        y_test = np_y_s[train:]

        # Calculating error for K values between 1 and 80
        for k in range(1, number_of_k + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            accuracy[k - 1] += (accuracy_score(y_test, y_pred))

    for j in range(0, number_of_k):
        accuracy[j] /= iterations
        # avg_accuracy[j] += accuracy[j]

    plt.plot(range(1, number_of_k + 1), accuracy, color='red', linestyle='solid')
    plt.show()

def knn_accuracy_table(model, labels):
    data = get_knn_accuracy(model, labels)
    for i, acc in enumerate(data):
        plt.bar(i, acc, color=colors[i])

    plt.xlabel("Schemas")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks([i for i, _ in enumerate(schemas)], schemas)
    plt.title("Binary classification with KNN")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("../data/FINAL_CSV.csv")
    texts, labels = utils.get_text_labels(df)
    tokenized_texts = utils.pre_process_data(texts)

    # training d2v model
    training_model_d2v(tokenized_texts)

    print("MODEL TRAINED")
    d2v_model = gensim.models.doc2vec.Doc2Vec.load('../models/schema-d2v-knn.model')

    # print(d2v_model.get_latest_training_loss())
    my_knn(d2v_model, labels)

    # for i, schema in enumerate(schemas):
    #     print(str(i) + ": " + schema)

    # knn_accuracy_plot(d2v_model, labels)
    #knn_accuracy_table(d2v_model, labels)

    # k_means_scatter_plot(d2v_model)
    #my_mlknn(d2v_model, labels)
    #knn_accuracy_table(d2v_model, labels)
    #mlknn_accuracy_plot(d2v_model, labels)
    #print_accuracy(get_knn_accuracy(d2v_model, labels))
    print("CHECK")
