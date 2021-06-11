import streamlit as st
from semanticsearch import search,utils,config
from semanticsearch.pretrained import get_model
from sentence_transformers import CrossEncoder
import numpy as np


st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache(allow_output_mutation=True)
def load_model():
    bi_encoder,index,documents=get_model(config.BI_ENCODER,config.INDEX,config.DATA)
    cross_encoder = CrossEncoder(config.CROSS_ENCODER)

    return bi_encoder,index,documents,cross_encoder


bi_encoder,index,documents,cross_encoder = load_model()

st.write(
    """
    # Semantic search with FAISS
    """
)

query = st.text_input('Query')


def make_prediction(query):
    value = search.search(query,index,bi_encoder,cross_encoder,documents)
    return {"Response_1": value["rank_1"],"Response_2": value["rank_2"],"Response_3": value["rank_3"]}


if not query:
    st.text("Please Enter your query")
else:
    st.write('The asked query is :::', query)
    prediction = make_prediction(query)
    st.json(prediction)
    st.success("Prediction made sucessful")
