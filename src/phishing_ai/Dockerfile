# Python 3.12: misma version con la que se generaron los .pkl.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Solo codigo de inferencia y modelos ya entrenados: el modelo no se entrena
# en el build (ver evaluation/comparar_modelos.py). Lo demas lo excluye
# .dockerignore.
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
