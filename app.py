from collections import OrderedDict
import streamlit as st
from streamlit.logger import get_logger
import experiments

LOGGER = get_logger(__name__)

EXPERIMENTS = OrderedDict(
    [
        ("—", (experiments.intro, None)),
        ("0525: Test adaptive testing mock UX",
         (experiments.test_at_mock_ux, None,),),
        ("0521: Explore trained models in 4 languages",
         (experiments.explore_language_models, None,),),
        ("0514: Explore trained English models",
         (experiments.explore_en_model, None,),),
        ("0507: Explore English quiz data",
         (experiments.explore_en_data, None,),),
    ]
)


def run():
    experiment_name = st.sidebar.selectbox(
        "Choose an experiment", list(EXPERIMENTS.keys()), 0)
    experiment = EXPERIMENTS[experiment_name][0]

    if experiment_name == "—":
        st.write("# Welcome to TomYo AI! 👋")
    else:
        st.markdown("# %s" % experiment_name)
        description = EXPERIMENTS[experiment_name][1]
        if description:
            st.write(description)
        for i in range(10):
            st.empty()

    experiment()


if __name__ == "__main__":
    run()
