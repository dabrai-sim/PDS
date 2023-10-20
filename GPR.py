import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

noise = 1
len_scale = 2.5

# Define your kernel_function and GPR functions as you have done

def kernel_function(x1, x2, len_scale):
    dist_sq = np.linalg.norm(x1 - x2) ** 2
    term = -1 / (2 * len_scale ** 2)
    return noise * np.exp(dist_sq * term)

def cov_matrix(x1, x2):
    n = x1.shape[0]
    m = x2.shape[0]
    cov_mat = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            cov_mat[i][j] = kernel_function(x1[i], x2[j], len_scale)
    return cov_mat

def GPR_train(trainX):
    K = cov_matrix(trainX, trainX)
    K_inv = np.linalg.inv(K + noise * np.identity(len(trainX)))
    return K, K_inv

def GPR_predict(trainX, trainY, testX):
    K1 = cov_matrix(trainX, testX)
    K2 = cov_matrix(testX, testX)
    K3 = K2 - np.matmul(K1.T, np.matmul(K_inv, K1)) + noise * np.identity(len(testX))

    mean_prediction = np.matmul(K1.T, np.matmul(K_inv, trainY))
    std_prediction = np.sqrt(np.diag(K3))

    return mean_prediction, std_prediction

st.title("Gaussian Process Regression with Streamlit")

# Collect user inputs for x and y
user_x = st.text_area("Enter x values (comma-separated):")
user_y = st.text_area("Enter y values (comma-separated):")

if st.button("Run GPR"):
    try:
        user_x = np.array([float(x) for x in user_x.split(",")])
        user_y = np.array([float(y) for y in user_y.split(",")])

        K, K_inv = GPR_train(user_x)

        st.write('Training completed')

        st.write('Testing started')
        mean_prediction, std_prediction = GPR_predict(user_x, user_y, user_x)
        st.write('Testing completed')

        st.write('Plotting...')
        fig, ax = plt.subplots()  # Create a Matplotlib figure
        ax.plot(user_x, user_y, color='black')
        ax.plot(user_x, mean_prediction, ls=':', lw=2, color='red')
        ax.fill_between(user_x, mean_prediction - 1.96 * std_prediction, mean_prediction + 1.96 * std_prediction, alpha=0.5, label=r"95% Confidence Interval")

        st.pyplot(fig)  # Pass the figure to st.pyplot

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
