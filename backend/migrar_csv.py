import pandas as pd
import mysql.connector


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,            
    "user": "root",          
    "password": "", 
    "database": "tcc"       
}

df = pd.read_csv("data/automation_data.csv")

conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS automation_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    tempo_execucao INT NOT NULL,
    freq_diaria INT NOT NULL,
    complexidade VARCHAR(20) NOT NULL,
    importancia TINYINT,
    urgencia TINYINT,
    ferramentas INT,
    volume INT,
    n_colaboradores INT,
    automatizar ENUM('sim', 'não') NOT NULL
);
""")

for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO automation_data 
        (tempo_execucao, freq_diaria, complexidade, importancia, urgencia, ferramentas, volume, n_colaboradores, automatizar)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, tuple(row))

conn.commit()
cursor.close()
conn.close()

print("✅ Migração concluída com sucesso!")

