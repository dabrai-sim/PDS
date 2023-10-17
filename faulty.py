import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"predictive_maintenance.csv")
del df['UDI']
del df['Type']
del df['Product ID']
label = df['Target'].to_numpy()
features = ['Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
features = df[features]

scaler = MinMaxScaler()
Xtransformed = scaler.fit_transform(features)
Xtrain = Xtransformed[:6000,:]
trainLabel = label[:6000,]
model = KNeighborsClassifier(n_neighbors=3)
model=model.fit(Xtrain,trainLabel)
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
