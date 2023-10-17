import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

feature1 = np.random.randint(60, 200, size=(10000, 1))
feature2 = np.random.randint(60, 300, size=(10000, 1))

feature = np.concatenate((feature1, feature2), axis=1)
label = np.random.randint(0, 2, size=(10000, 1))
label = label.reshape((label.shape[0],))
model = KNeighborsClassifier(n_neighbors=5)
model.fit(feature, label)


def main():
    st.title('Diabetes Prediction with kNN')
    # st.write('This app uses k-Nearest Neighbors (kNN) to classify Iris flowers into three species.')

    # Collect input features from the user
    # Blood_Pressure = st.slider('Blood Pressure', 60, 200, 80)
    Blood_Pressure = st.number_input('Blood Pressure')
    Blood_Glucose = st.slider('Blood Glucose', 60, 300, 140)

    # Create a feature array with the user's input
    features = np.array([[Blood_Pressure, Blood_Glucose]])

    # Make predictions using the kNN model
    prediction = model.predict(features)
    if prediction == 0:
        predicted_label = 'Non-Diabetic'
    else:
        predicted_label = 'Diabetic'

        # Display the prediction
    st.write(f'The patient is : {predicted_label}')


if __name__ == '__main__':
    main()
