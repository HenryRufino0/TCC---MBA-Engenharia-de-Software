from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from automation_tree import AutomationTree
from utils import load_automation_data
from models import TaskInput
import pandas as pd
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tree = AutomationTree()

@app.post("/avaliar")
def avaliar_automacao(input: TaskInput):
    try:
        resultado = tree.predict(
            frequencia=input.frequencia,
            tempo=input.tempo,
            complexidade=input.complexidade.lower(),
            importancia=input.importancia.lower(),
            urgencia=input.urgencia.lower(),
            ferramentas=input.ferramentas,
            volume=input.volume,
            colaboradores=input.colaboradores
        )
        return {"automatizar": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def treinar_base_padrao():
    try:
        df = load_automation_data("data/automation_data.csv")
        tree.train(df)
    except Exception as e:
        print(f"Erro ao carregar dados padrão: {e}")

@app.get("/metricas")
def metricas():
    if not tree.is_trained or not tree.metrics():
        return JSONResponse(status_code=503, content={"detail": "Modelo não treinado ou métricas indisponíveis."})
    return tree.metrics()