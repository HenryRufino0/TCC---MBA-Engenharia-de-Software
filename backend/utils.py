import pandas as pd
from io import StringIO

def load_automation_data(path):
    df = pd.read_csv(path)
    required_columns = {
        "frequencia", "tempo", "complexidade", "importancia", "urgencia",
        "ferramentas", "volume", "colaboradores", "automatizar"
    }
    if not required_columns.issubset(set(df.columns)):
        raise ValueError(f"CSV deve conter as colunas: {', '.join(required_columns)}")
    return df

def default_data():
    csv_data = """
frequencia,tempo,complexidade,importancia,urgencia,ferramentas,volume,colaboradores,automatizar
5,10,baixa,sim,sim,4,8,2,sim
10,120,baixa,sim,nao,7,50,8,sim
3,20,media,sim,nao,6,30,4,sim
1,45,alta,nao,sim,3,10,1,nao
7,15,media,sim,sim,5,20,3,sim
2,60,alta,nao,nao,2,15,1,nao
8,5,baixa,sim,sim,4,8,2,sim
6,100,media,sim,nao,7,50,7,sim
4,12,baixa,nao,sim,3,6,1,nao
"""
    return pd.read_csv(StringIO(csv_data.strip()))
