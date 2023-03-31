import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

if "x0" not in st.session_state:
    st.session_state.x0 = None
if "x1" not in st.session_state:
    st.session_state.x1 = None
if "Xbar" not in st.session_state:
    st.session_state.Xbar = None
if "y" not in st.session_state:
    st.session_state.y = None
if "w_init" not in st.session_state:
    st.session_state.w_init = None
if "df" not in st.session_state:
    st.session_state.df = None
if "log" not in st.session_state:
    st.session_state.log = None
if "w" not in st.session_state:
    st.session_state.w = None


class Log:
    def __init__(self) -> None:
        self.w = list()
        self.iter = 0
    def add(self, w):
        self.w.append(w)
        self.iter += 1
    def get(self):
        return self.w
    

def predict(w, X):
    '''
    Predicts the labels of the data points in X using the weights w.
    X: a 2D numpy array of shape (n, d) where each row is a data point.
    w: a 1D numpy array of shape (d,).
    '''
    return np.sign(X.dot(w))

def perceptron(X, y, w_init, log: Log(), learning_rate = 0.1):
    w = w_init
    while True:
        if log is not None:
            log.add(w)
        pred = predict(w, X)
        mis_idx = np.where(np.equal(pred, y) == False)[0]
        num_mis = mis_idx.shape[0]
        if num_mis == 0:
            if log is not None:
                log.add(w)
            return w
        random_id = np.random.choice(mis_idx, 1) [0]
        w = w + learning_rate * y[random_id] * X[random_id]
        

def initialize(N = 10):
    means = [[-1, 0], [1, 0]]
    cov = [[0.3, 0.2], [0.2, 0.3]]
    x0 = np.random.multivariate_normal(means[0], cov, N)
    x1 = np.random.multivariate_normal(means[1], cov, N)

    X = np.concatenate((x0, x1), axis = 0)
    y = np.concatenate((np.ones(N), -1*np.ones(N)), axis = 0)

    Xbar = np.concatenate((np.ones((2*N, 1)), X), axis = 1)
    w_init = np.random.randn(Xbar.shape[1])
    
    st.session_state.x0 = x0
    st.session_state.x1 = x1
    st.session_state.Xbar = Xbar
    st.session_state.y = y
    st.session_state.w_init = w_init

    df = pd.DataFrame(Xbar, columns=["x0", "x1", "x2"])
    df["label"] = y
    df.drop(df.columns[0], axis=1, inplace=True)

    st.session_state.df = df


    return Xbar, y, w_init






st.set_page_config(layout="wide", page_title="SVM", page_icon="✖️")
st.title("Perceptron Learning Algorithm (PLA) Demonstration")

st.button("Click here to generate the dataset", on_click=initialize, args=())
if st.session_state.Xbar is not None:
    cols = st.columns(2)
    cols[0].dataframe(st.session_state.df)
    cols[0].success("The dataset is generated")
    sns.scatterplot(x=st.session_state.df['x1'], y=st.session_state.df['x2'], hue=st.session_state.df["label"])
    cols[1].pyplot()
if st.session_state.Xbar is not None:
    learning_rate = cols[0].number_input("Learning rate", value=0.1, step=0.1, format="%.1f")
if st.button("Implement Perceptron Learning Algorithm (PLA)"):
    st.session_state.log = None
    # log = Log()
    # w = perceptron(Xbar, y, w_init, log)
    perceptron_log = Log()
    w = perceptron(st.session_state.Xbar , st.session_state.y, st.session_state.w_init, perceptron_log, learning_rate = learning_rate)

    st.session_state.w = w
    st.session_state.log = perceptron_log

def plot(iter):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.scatter(st.session_state.x0[:, 0], st.session_state.x0[:, 1], c='red', edgecolors='k')
    ax.scatter(st.session_state.x1[:, 0], st.session_state.x1[:, 1], c='blue', edgecolors='k')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    x = np.linspace(-2, 2, 100)
    y = -(st.session_state.log.get()[iter][0] + st.session_state.log.get()[iter][1]*x)/st.session_state.log.get()[iter][2]
    ax.plot(x, y, 'k')
    return fig


if st.session_state.log is not None:
    col1, col2 = st.columns(2)
    with col1:
        # st.write("The final weight vector is: ", st.session_state.w)
        st.write("The number of iterations is: ", len(st.session_state.log.get()) - 1)
        plt.figure(figsize=(10, 12))
        sns.scatterplot(x=st.session_state.df['x1'], y=st.session_state.df['x2'], hue=st.session_state.df["label"])
        plt.x_lim = (-2, 2)
        plt.y_lim = (-2, 2)
        x = np.linspace(-2, 2, 100)
        for iter, w in enumerate(st.session_state.log.get()):
            y = -w[0]/w[2] - w[1]/w[2]*x
            sns.lineplot(x = x, y = y, label="iter: " + str(iter))
        # y = -w[0]/w[2] - w[1]/w[2]*x
        # plt.plot(x, y)
        plt.legend()
        st.pyplot()
    with col2:
        iter = st.slider("Iteration", min_value=0, max_value=st.session_state.log.iter - 1, value=0)
        st.pyplot(plot(iter))
    
        st.write("The weights of the model are: ")
        

