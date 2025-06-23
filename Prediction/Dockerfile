FROM python:3.10-slim

# Dépendances système
RUN apt-get update && apt-get install -y libaio1 unzip && rm -rf /var/lib/apt/lists/*

# Copier le client Oracle dans le conteneur
COPY oracle/instantclient_19_26 /opt/oracle/instantclient_19_26

# Configurer les variables d'environnement
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_19_26
ENV PATH=$LD_LIBRARY_PATH:$PATH

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api_catboost.py"]
