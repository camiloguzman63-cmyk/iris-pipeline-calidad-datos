FROM swaggerapi/swagger-ui:v5.17.14 AS swagger
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py approved.json data_contract.json ./
COPY --from=swagger /usr/share/nginx/html/swagger-ui-bundle.js /usr/share/nginx/html/swagger-ui.css ./static/
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
