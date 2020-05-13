"""
App: TomYo AI - User English Level Prediction
Author: Gereltuya
Credits: Noah Saunders, Marc Skov Madsen, Jason Brownlee
Source Code/Article(s): Awesome Streamlit (https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/gallery/iris_classification/iris.py), Machine Learning Mastery (https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)
Source Data: https://tomyo.mn/
"""

import pathlib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from keras.utils import np_utils


DATA_CSV_FILE = "data/UEHAugmented.csv?token=AHDXJ7OOJOZSGEIIGUQ2Z4S6YU6EM"
LOCAL_ROOT = pathlib.Path(__file__).parent
GITHUB_ROOT = (
    "https://raw.githubusercontent.com/qerelt/tomyo-ai/master/"
)

def main():
    """## Main function of TomYo AI - User English Level Prediction App

    Run this to run the app.
    """
    st.title("TomYo AI - User English Level Classifier")
    st.header("Data Exploration")

    source_df = read_data_csv()
    st.subheader("Source Data")
    if st.checkbox("Show Source Data"):
        st.write(source_df)

    selected_level_df = select_level(source_df)
    if not selected_level_df.empty:
        show_scatter_plot(selected_level_df)
        show_histogram_plot(selected_level_df)
    else:
        st.info("Please select one of more levels above for further exploration.")

    show_machine_learning_model(source_df)


def select_level(source_df: pd.DataFrame) -> pd.DataFrame:
    """## Component for selecting one of more levels for exploration

    Arguments:
        source_df {pd.DataFrame} -- The source dataframe

    Returns:
        pd.DataFrame -- A sub dataframe having data for the selected levels
    """
    selected_levels = st.multiselect(
        "Select source data levels for further exploration below",
        source_df["userLevelByHuman"].unique(),
    )
    selected_level_df = source_df[(
        source_df["userLevelByHuman"].isin(selected_levels))]
    if selected_levels:
        st.write(selected_level_df)
    return selected_level_df


def show_scatter_plot(selected_level_df: pd.DataFrame):
    """## Component to show a scatter plot of two features for the selected levels

    Arguments:
        selected_level_df {pd.DataFrame} -- A DataFrame with the same columns as the
            source_df source data dataframe
    """
    st.subheader("Scatter plot")
    feature_x = st.selectbox("Which feature on x?",
                             selected_level_df.columns[0:9])
    feature_y = st.selectbox("Which feature on y?",
                             selected_level_df.columns[0:9])

    fig = px.scatter(selected_level_df, x=feature_x,
                     y=feature_y, color="userLevelByHuman")
    st.plotly_chart(fig)


def show_histogram_plot(selected_level_df: pd.DataFrame):
    """## Component to show a histogram of the selected levels and a selected feature

    Arguments:
        selected_level_df {pd.DataFrame} -- A DataFrame with the same columns as the
            source_df source data dataframe
    """
    st.subheader("Histogram")
    feature = st.selectbox("Which feature?", selected_level_df.columns[0:9])
    fig2 = px.histogram(selected_level_df, x=feature,
                        color="userLevelByHuman", marginal="rug")
    st.plotly_chart(fig2)


def show_machine_learning_model(source_df: pd.DataFrame):
    """Component to show the performance of a DL model trained on the source data set

    Arguments:
        source_df {pd.DataFrame} -- The source data set

    Raises:
        NotImplementedError: Raised if a not supported model is selected
    """
    st.header("Model Exploration")
    features = source_df[
        ["A1", "A2", "B1", "B2", "C1", "C2", "TOEFL", "IELTS", "SAT"]
    ].values
    labels = source_df["userLevelByHuman"].values

    modelVersions = ["V1", "V2"]
    modelVersion = st.selectbox("Which version?", modelVersions)

    if modelVersion == "V1":
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, train_size=0.9, random_state=7
        )
        y_test_v = y_test.tolist()
    elif modelVersion == "V2":
        encoder = LabelEncoder()
        encoder.fit(labels)
        encoded_labels = encoder.transform(labels)
        onehot_labels = np_utils.to_categorical(encoded_labels)
        x_train, x_test, y_train, y_test = train_test_split(features, onehot_labels, train_size=0.9, random_state=7)
        y_test_v = [sum([y_test[j][i]*i for i in range(5)]) for j in range(len(y_test))]


    lossFunctions = {"V1": "binary_crossentropy", "V2": "categorical_crossentropy"}
    optimizers = {"V1": "rmsprop", "V2": "adam"}
    modelStructure = "model/model{0}.json".format(modelVersion)
    modelWeights = "model/model{0}.h5".format(modelVersion)
    json_file = open(modelStructure, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(modelWeights)
    model.compile(loss=lossFunctions[modelVersion], optimizer=optimizers[modelVersion], metrics=['accuracy'])
    acc = model.evaluate(x_test, y_test, verbose=0)[1]*100
    st.write("Accuracy: ", acc.round(2))
    pred_model = model.predict_classes(x_test)
    cm_model = confusion_matrix(y_test_v, pred_model)
    st.write("Confusion matrix: ", cm_model)
    predictions = model.predict_classes(x_test)
    predictions_dict = []
    for i in range(len(predictions)):
        predictions_dict.append({"predicted": predictions[i], "expected": y_test_v[i], "isCorrect": predictions[i]==y_test_v[i], "input": x_test[i].tolist()})
    predictions_df = pd.DataFrame.from_dict(predictions_dict)
    st.write("Predictions on test data: ", predictions_df)


@st.cache
def read_data_csv() -> pd.DataFrame:
    """## source data dataframe

    Returns:
        pd.DataFrame -- A dataframe with the source data
    """
    # return pd.read_csv(LOCAL_ROOT / DATA_CSV_FILE)
    return pd.read_csv("data/UEHAugmented.csv")


main()
