import pandas as pd
import mysql.connector

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "",
    "database": "tcc"
}

def load_automation_data():
    conn = mysql.connector.connect(**DB_CONFIG)
    query = """
        SELECT tempo_execucao, freq_diaria, complexidade, importancia, urgencia, 
               ferramentas, volume, n_colaboradores, automatizar
        FROM automation_data
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Renomear para os nomes que o modelo espera
    df = df.rename(columns={
        "tempo_execucao": "tempo",
        "freq_diaria": "frequencia",
        "n_colaboradores": "colaboradores"
    })

    return df
