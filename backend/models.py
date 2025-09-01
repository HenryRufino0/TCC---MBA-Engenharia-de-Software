from pydantic import BaseModel

class TaskInput(BaseModel):
    frequencia: int
    tempo: float
    complexidade: str
    importancia: str
    urgencia: str
    ferramentas: int
    volume: int
    colaboradores: int
