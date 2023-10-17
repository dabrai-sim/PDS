import streamlit as st
import numpy as np
import pandas as pd
import math,time
noise = 1
len_scale = 2.5

df = pd.read_csv(r"G:\Users\dabra\downloads\air.csv")
def kernel_function(x1, x2, len_scale):
  dist_sq = np.linalg.norm(x1-x2)**2
  term = -1/(2*len_scale**2)
  return noise*np.exp(dist_sq*term)
def cov_matrix(x1, x2):
  n = x1.shape[0]
  m = x2.shape[0]
  cov_mat = np.empty((n, m))

  for i in range(n):
    for j in range(m):
      cov_mat[i][j] = kernel_function(x1[i], x2[j], len_scale)
  return cov_mat
def GPR_train(trainX, trainY):
  K = cov_matrix(trainX,trainX)
  K_inv = np.linalg.inv(K+noise*np.identity(len(trainX)))
  return K, K_inv
def GPR_predict(trainX, trainY, testX):
  K1 = cov_matrix(trainX, testX)
  K2 = cov_matrix(testX, testX)
  K3 = K2-np.matmul(K1.T, np.matmul(K_inv,K1))+noise*np.identity(len(testX))

  mean_prediction = np.matmul(K1.T,np.matmul(K_inv,trainY))
  std_prediction = np.sqrt(np.diag(K3))

  return mean_prediction,std_prediction
trainX = np.linspace(0, 10, num=1000)
trainY = trainX*np.sin(trainX)

testX = np.linspace(0,10,num=1000)
testY = testX*np.sin(testX)

K, K_inv = GPR_train(trainX, trainY)
mean_prediction,std_prediction=GPR_predict(trainX,trainY,testX)

def main():
    st.title('Machine Fault Detection with kNN')
    Airtemperature= st.number_input('Air Temperature')
    Processtemperature= st.number_input('Process Temperature')
    Rotationalspeed= st.number_input('Rotational speed')
    Torque= st.number_input('Torque')
    Toolwear = st.number_input('Toolwear')

    # Create a feature array with the user's input
    features1 = np.array([[Airtemperature, Processtemperature, Rotationalspeed, Torque, Toolwear]])

    # Make predictions using the kNN model
    prediction = model.predict(features1)
    if prediction == 0:
        predicted_label = 'Not Faulty'
    else:
        predicted_label = 'Faulty'

        # Display the prediction
    st.write(f'The Machine is : {predicted_label}')


if __name__ == '__main__':
    main()
