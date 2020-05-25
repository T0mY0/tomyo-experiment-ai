import inspect
import textwrap
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from keras.utils import np_utils
from random import randrange, shuffle, random
import numpy as np


def intro():

    st.markdown(
        """

        This app is for showcasing all kinds of experiments the TomYo AI/Data team is running at any point in time.

        Please keep in mind that these are all very experimental and things are done in an ad hoc manner.

        You can also check out [the source code](https://bitbucket.org/tomyomn/tomyo-ai/src/master/) on Bitbucket.

        **👈 Select an experiment from the dropdown on the left**
    """
    )


@st.cache
def read_data(data_filepath) -> pd.DataFrame:
    return pd.read_csv(data_filepath)


def explore_data(data_filepath: str):

    def select_level(source_df: pd.DataFrame) -> pd.DataFrame:
        selected_levels = st.multiselect(
            "Select source data levels for further exploration below",
            source_df["userLevelByHuman"].unique(),
        )
        selected_level_df = source_df[(
            source_df["userLevelByHuman"].isin(selected_levels))]
        if selected_levels:
            st.subheader("2. Filtered source data")
            st.write(selected_level_df)
        return selected_level_df

    def show_scatter_plot(selected_level_df: pd.DataFrame):
        st.subheader("3. Scatter plot")
        feature_x = st.selectbox("Which feature on x?",
                                 selected_level_df.columns)
        feature_y = st.selectbox("Which feature on y?",
                                 selected_level_df.columns)

        fig = px.scatter(selected_level_df, x=feature_x,
                         y=feature_y, color="userLevelByHuman")
        st.plotly_chart(fig)

    def show_histogram_plot(selected_level_df: pd.DataFrame):
        st.subheader("4. Histogram")
        feature = st.selectbox(
            "Which feature?", selected_level_df.columns[:])
        fig2 = px.histogram(selected_level_df, x=feature,
                            color="userLevelByHuman", marginal="rug")
        st.plotly_chart(fig2)

    st.subheader("1. Source data")
    source_df = read_data(data_filepath)
    if st.checkbox("Show data"):
        st.write(source_df)

    selected_level_df = select_level(source_df)
    if not selected_level_df.empty:
        show_scatter_plot(selected_level_df)
        show_histogram_plot(selected_level_df)
    else:
        st.info("Please select one of more levels above for further exploration.")

    st.subheader("5. Code")
    if st.checkbox("Show code"):
        sourcelines, _ = inspect.getsourcelines(explore_data)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


def explore_en_data():
    en_data_filepath = "data/20200506/UEHAugmented.csv"
    explore_data(en_data_filepath)


def explore_en_model():

    en_data_filepath = "data/20200506/UEHAugmented.csv"
    source_df = read_data(en_data_filepath)
    num_features = 9
    feature_values = source_df[source_df.columns[:num_features]].values
    num_labels = 5
    label_values = source_df[source_df.columns[num_features]].values

    st.subheader("1. Model version")
    model_versions = ["V2", "V1"]
    model_version = st.selectbox("Choose model version below", model_versions)

    if model_version == "V1":
        x_train, x_test, y_train, y_test = train_test_split(
            feature_values, label_values, train_size=0.9, random_state=7)
        y_test_v = y_test.tolist()
    elif model_version == "V2":
        encoder = LabelEncoder()
        encoder.fit(label_values)
        encoded_labels = encoder.transform(label_values)
        onehot_labels = np_utils.to_categorical(encoded_labels)
        x_train, x_test, y_train, y_test = train_test_split(
            feature_values, onehot_labels, train_size=0.9, random_state=7)
        y_test_v = [sum([y_test[j][i] * i for i in range(num_labels)])
                    for j in range(len(y_test))]

    loss_functions = {"V1": "binary_crossentropy",
                      "V2": "categorical_crossentropy"}
    optimizers = {"V1": "rmsprop", "V2": "adam"}
    model_structure = "model/20200506/model{0}.json".format(model_version)
    model_weights = "model/20200506/model{0}.h5".format(model_version)
    json_file = open(model_structure, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(model_weights)
    model.compile(loss=loss_functions[model_version],
                  optimizer=optimizers[model_version], metrics=['accuracy'])
    acc = model.evaluate(x_test, y_test, verbose=0)[1] * 100
    st.subheader("2. Loading trained model")
    st.write("Accuracy: ", acc.round(2))

    pred_model = model.predict_classes(x_test)
    cm_model = confusion_matrix(y_test_v, pred_model)
    st.subheader("3. Predicting classes")
    st.write("Confusion matrix:", cm_model)

    st.write("Input type:", type(x_test[0]))
    st.write("Input shape:", x_test[0].shape)
    st.write("Input sample:", x_test[0])

    if model_version == "V1":
        predictions = model.predict_classes(x_test)[:, 0]
    elif model_version == "V2":
        predictions = model.predict_classes(x_test)
    predictions_dict = []
    for i in range(len(predictions)):
        features = source_df.columns[:num_features]
        labels_to_levels = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C"}
        predictions_elem = {}
        predictions_elem["predicted"] = labels_to_levels[predictions[i]]
        predictions_elem["expected"] = labels_to_levels[y_test_v[i]]
        predictions_elem["isCorrect"] = True if predictions[i] == y_test_v[i] else False
        for j in range(num_features):
            predictions_elem[features[j]] = x_test[i].round(3).tolist()[j]
        predictions_dict.append(predictions_elem)
    predictions_df = pd.DataFrame.from_dict(predictions_dict)
    st.write("Predictions on test data: ", predictions_df)

    st.subheader("4. Code")
    if st.checkbox("Show code"):
        sourcelines, _ = inspect.getsourcelines(explore_en_model)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


def explore_language_models():

    st.subheader("1. Language")
    languages = ["en", "de", "ja", "ko"]
    selected_language = st.selectbox(
        "Choose language below (en:English, de:German, ja:Japanese, ko:Korean)", languages)

    language_data_filepath = "data/20200519/{0}/userHistoryAugmented.csv".format(
        selected_language)
    source_df = read_data(language_data_filepath)
    num_features = {"en": 9, "de": 5, "ja": 5, "ko": 5}[selected_language]
    feature_values = source_df[source_df.columns[:num_features]].values
    num_labels = 5
    label_values = source_df[source_df.columns[num_features]].values

    encoder = LabelEncoder()
    encoder.fit(label_values)
    encoded_labels = encoder.transform(label_values)
    onehot_labels = np_utils.to_categorical(encoded_labels)
    x_train, x_test, y_train, y_test = train_test_split(
        feature_values, onehot_labels, train_size=0.9, random_state=7)
    y_test_v = [sum([y_test[j][i] * i for i in range(num_labels)])
                for j in range(len(y_test))]

    model_structure = "model/20200519/{0}/modelV2.json".format(
        selected_language)
    model_weights = "model/20200519/{0}/modelV2.h5".format(selected_language)
    json_file = open(model_structure, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(model_weights)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=['accuracy'])
    acc = model.evaluate(x_test, y_test, verbose=0)[1] * 100
    st.subheader("2. Loading trained model")
    st.write("Accuracy: ", acc.round(2))

    pred_model = model.predict_classes(x_test)
    cm_model = confusion_matrix(y_test_v, pred_model)
    st.subheader("3. Predicting classes")
    st.write("Confusion matrix:", cm_model)

    predictions = model.predict_classes(x_test)
    st.write("Input type:", type(x_test[0]))
    st.write("Input shape:", x_test[0].shape)
    st.write("Input sample:", x_test[0])
    predictions_dict = []
    for i in range(len(predictions)):
        features = source_df.columns[:num_features]
        labels_to_levels = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C"}
        predictions_elem = {}
        predictions_elem["predicted"] = labels_to_levels[predictions[i]]
        predictions_elem["expected"] = labels_to_levels[y_test_v[i]]
        predictions_elem["isCorrect"] = True if predictions[i] == y_test_v[i] else False
        for j in range(num_features):
            predictions_elem[features[j]] = x_test[i].round(3).tolist()[j]
        predictions_dict.append(predictions_elem)
    predictions_df = pd.DataFrame.from_dict(predictions_dict)
    st.write("Predictions on test data: ", predictions_df)

    st.subheader("4. Code")
    if st.checkbox("Show code"):
        sourcelines, _ = inspect.getsourcelines(explore_language_models)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


def compile_de_model():

    model_structure = "model/20200519/de/modelV2.json"
    model_weights = "model/20200519/de/modelV2.h5"
    json_file = open(model_structure, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(model_weights)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=['accuracy'])
    return model


def test_at_mock_ux():

    num_q = 10
    u_answers = {0: {"level": "A1", "answer": ""},
                 1: {"level": "A1", "answer": ""},
                 2: {"level": "A2", "answer": ""},
                 3: {"level": "A2", "answer": ""},
                 4: {"level": "B1", "answer": ""},
                 5: {"level": "B1", "answer": ""},
                 6: {"level": "B2", "answer": ""},
                 7: {"level": "B2", "answer": ""},
                 8: {"level": "C", "answer": ""},
                 9: {"level": "C", "answer": ""}}
    u_scores = {"A1": {"answered": 0, "correct": 0},
                "A2": {"answered": 0, "correct": 0},
                "B1": {"answered": 0, "correct": 0},
                "B2": {"answered": 0, "correct": 0},
                "C": {"answered": 0, "correct": 0}}
    u_recommendations = {"A1": (8, 2, 0, 0, 0),
                         "A2": (2, 6, 2, 0, 0),
                         "B1": (0, 2, 6, 2, 0),
                         "B2": (0, 0, 2, 6, 2),
                         "C": (0, 0, 0, 2, 8),
                         "undefined": (2, 2, 2, 2, 2)}
    labels_to_levels = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C"}

    st.write("We will be testing mock user experience of adaptive testing using real (97% accurate) deep learning model trained on TomYo German quiz data as of 20200519.")
    st.subheader("1. Ready, set, action, go!")
    st.write("Number of questions:", num_q)
    for i in range(num_q):
        u_answers[i]["answer"] = st.selectbox("Question no. {0}, level {1}".format(i, u_answers[i]["level"]), [
            "Not answered", "Correct", "Incorrect"])
        if u_answers[i]["answer"] == "Correct":
            u_scores[u_answers[i]["level"]]["answered"] += 1
            u_scores[u_answers[i]["level"]]["correct"] += 1
        elif u_answers[i]["answer"] == "Incorrect":
            u_scores[u_answers[i]["level"]]["answered"] += 1

    st.subheader("2. Your current scores:")
    u_scores_list = []
    for level in u_scores:
        score = 0.0 if u_scores[level]["answered"] == 0 else round(
            u_scores[level]["correct"] / u_scores[level]["answered"], 2)
        u_scores_list.append(score)
        st.write(level, ":", u_scores[level]["correct"],
                 "/", u_scores[level]["answered"], "=", score)
    u_scores_ndarray = np.array([u_scores_list])

    st.subheader("3. Predicting your language level:")
    st.write("Formatted input to the model:", u_scores_ndarray)
    u_level = "undefined"
    if st.button("Update my level  👈"):
        model = compile_de_model()
        predictions = model.predict_classes(u_scores_ndarray)
        st.balloons()
        st.write("Formatted output of the model:", predictions)
        u_level = labels_to_levels[predictions[0]]
        st.write("Your updated level is:", u_level)

    st.subheader(
        "4. Recommending new set of questions based on your updated level:")
    st.write("Your current level is:", u_level)
    st.write(
        "Now you will get these # of recommended questions and repeat the steps 1 -> 4 again:")
    for i in range(5):
        st.write(labels_to_levels[i], u_recommendations[u_level][i])

    st.subheader("5. Code")
    if st.checkbox("Show code"):
        sourcelines, _ = inspect.getsourcelines(test_at_mock_ux)
        st.code(textwrap.dedent("".join(sourcelines[1:])))
