# Clasificación de flores Iris con control de calidad de datos

Seminario de Grado II · Actividad 2.1 · Universidad Santo Tomás, Seccional Tunja

El proyecto clasifica una flor como setosa, versicolor o virginica a partir de cuatro medidas. Un único comando prepara los datos, comprueba su calidad, entrena un bosque aleatorio y comprueba si alcanza la exactitud mínima de 0,80 en datos reservados.

## Reproducir desde cero

1. Descomprima el proyecto y abra una terminal en la carpeta que contiene este README y run_pipeline.py. Se requiere Python 3.12, conexión a internet durante la instalación y permiso de escritura en el directorio. El conjunto Iris se incluye con scikit-learn.
2. Cree y active un entorno virtual. En Windows (cmd.exe):

       py -3.12 -m venv .venv
       .venv\Scripts\activate.bat

   En macOS o Linux:

       python3.12 -m venv .venv
       source .venv/bin/activate

3. Instale las dependencias directas. Para replicar las versiones transitivas del entorno de referencia Linux, use en su lugar requirements-linux.lock.

       python -m pip install -r requirements.txt

4. Ejecute el flujo completo:

       python run_pipeline.py

   El proceso devuelve 0 si termina correctamente. Cada ejecución tiene un directorio independiente en runs/. La consola indica la ruta concreta. En caso de error, consulte pipeline.log en esa ruta. El flujo se detiene al fallar una etapa. En MLflow, una ejecución finalizada correctamente figura como FINISHED.

## Datos y contrato

Se usa la versión corregida de Iris distribuida en scikit-learn 1.8.0 (150 observaciones, 50 por especie). La columna species solo se usa como etiqueta al entrenar y evaluar. La petición de inferencia debe ser un objeto JSON con exactamente los cuatro campos de la tabla. Los intervalos son inclusivos y delimitan el dominio de este ejercicio con base en el conjunto de referencia.

| Campo | Medición en centímetros | Intervalo |
|---|---|---|
| sepal_length | Longitud del sépalo | 4,3 a 7,9 |
| sepal_width | Ancho del sépalo | 2,0 a 4,4 |
| petal_length | Longitud del pétalo | 1,0 a 6,9 |
| petal_width | Ancho del pétalo | 0,1 a 2,5 |

data_contract.json define las reglas y contiene ambos ejemplos. En JSON se usa punto decimal. El validador rechaza campos ausentes o adicionales, texto, booleanos, nulos y valores fuera del intervalo. Al leer CSV, convierte el texto numérico a su valor y comprueba la cabecera y la clase. Un error identifica el archivo, la fila y el campo cuando corresponde.

## Etapas y archivos

| Archivo | Responsabilidad |
|---|---|
| run_pipeline.py | Ejecuta las cuatro etapas, conserva la bitácora y registra el experimento en MLflow. |
| src/prepare.py | Separa por identificador de fila con semilla 42 y genera train.csv y test.csv estratificados. |
| src/validate_data.py | Aplica el contrato con la biblioteca estándar de Python. |
| src/train.py | Ajusta 100 árboles con semilla 42 sobre 120 filas y evalúa 30 filas reservadas. |
| tests/test_model.py | Ejecuta diez pruebas de esquema, límites, particiones, artefactos y desempeño. |
| predict.py | Valida una nueva observación y devuelve la especie predicha. |

## Resultados y trazabilidad

La ejecución de referencia en Python 3.12.14 acertó en 27 de 30 flores (0,90). Clasificó correctamente las diez setosa, nueve versicolor y ocho virginica. Los tres errores correspondieron a confusiones entre las dos últimas especies. Con solo diez flores de prueba por especie, un error cambia diez puntos porcentuales el acierto de esa clase. Se usa una sola partición, sin ajuste de hiperparámetros sobre la prueba ni validación externa.

En cada directorio de ejecución quedan los dos CSV, model.pkl, metrics.json y pipeline.log. La matriz de confusión se lee por filas de especie real y columnas de predicción, en el orden de clases indicado en metrics.json. MLflow conserva parámetros, exactitud, copias de los artefactos de la ejecución aprobada y huellas SHA-256 de las particiones y el contrato. Para consultar las ejecuciones, desde la carpeta raíz:

    mlflow ui --backend-store-uri ./mlruns --port 5000

Abra http://127.0.0.1:5000 y seleccione el experimento Iris. Si el puerto está ocupado, elija otro y ajuste la URL.

## Comprobar una inferencia

Sustituya RUTA_DE_EJECUCION por la ruta impresa en la primera etapa. Use el modelo generado en su propia ejecución aprobada.

    python predict.py RUTA_DE_EJECUCION ejemplos/valida.json
    python predict.py RUTA_DE_EJECUCION ejemplos/invalida.json

El primer ejemplo devuelve setosa. El segundo termina con código 1 e indica el tipo incorrecto, el nulo y la medida negativa. Para repetir las pruebas sobre los archivos de esa ejecución:

    python tests/test_model.py RUTA_DE_EJECUCION

## Fuentes

- Contreras Ortiz, M. S., Puerto Moreno, S. A., Suárez Rojas, A. L., y Vela Beltrán, D. A. (2026). Métodos de investigación en Data Science. Ediciones USTA Tunja. Capítulo 2, apartado 2.7, pp. 111–113.
- Contreras Ortiz, M. S. (2026, 19 de noviembre). Estandarización de pipelines y control de calidad de datos. Seminario de Grado II, sesión 4.
- Datos: [Iris en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html). Conjunto original: [UCI Machine Learning Repository](https://doi.org/10.24432/C56C76).
