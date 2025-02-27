FROM python:3.12.9

WORKDIR /workdir

COPY requirements.txt .

# Instalar dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Configurar directorios de importación por defecto
ENV PYTHONPATH="/workdir/src:/workdir/dev"

# Exponer el puerto en el que corre la app (Flask usa 5000 por defecto)
EXPOSE 5000

# Ejecutar la aplicación Flask
CMD ["python", "app.py"]
