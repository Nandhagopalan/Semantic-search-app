# Fast APi Imports
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Package imports
from semanticsearch import search,utils,config
from semanticsearch.pretrained import get_model
from sentence_transformers import CrossEncoder

import numpy as np


def load_model():
    bi_encoder,index,documents=get_model(config.BI_ENCODER,config.INDEX,config.DATA)
    cross_encoder = CrossEncoder(config.CROSS_ENCODER)

    return bi_encoder,index,documents,cross_encoder


bi_encoder,index,documents,cross_encoder = load_model()

# Initializing the Fast API
app = FastAPI(
    title="Semantic Search",
    version="0.1.0",
    description="Answering your queries related to covid from a corpus of research papers",
)

# Adding CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/answer")
async def rank_answer(query:str):
    value = search.search(query,index,bi_encoder,cross_encoder,documents)
    return {"Response_1": value["rank_1"],"Response_2": value["rank_2"],"Response_3": value["rank_3"]}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.0", port=8003, reload=True)
