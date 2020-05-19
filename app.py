from collections import OrderedDict
import streamlit as st
from streamlit.logger import get_logger
import experiments

LOGGER = get_logger(__name__)

EXPERIMENTS = OrderedDict(
    [
        ("—", (experiments.intro, None)),
        ("Explore English quiz data",(experiments.explore_en_data,None,),),
        ("Explore trained English models",(experiments.explore_en_model,None,),),
    ]
)


def run():
    experiment_name = st.sidebar.selectbox("Choose an experiment", list(EXPERIMENTS.keys()), 0)
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
