# CapstoneProject
Validación Cruzada con XGBoost y LSTM
Este es un ejemplo de validación cruzada utilizando los algoritmos XGBoost y LSTM para la clasificación de datos meteorológicos. El código realiza el procesamiento de datos, construye los modelos y evalúa su rendimiento utilizando métricas como precisión, error absoluto medio, error cuadrado medio, recuperación y medida F.
# Requisitos
•	Python 3.x
•	Bibliotecas de Python: numpy, pandas, matplotlib, seaborn, xgboost, scikit-learn, tensorflow
# Instalación
1.Clona el repositorio o descarga el código fuente en tu máquina local.
2.Asegúrate de tener Python 3.x instalado. Puedes descargarlo desde el sitio web oficial de Python: https://www.python.org/downloads/
3.Instala las bibliotecas de Python necesarias ejecutando el siguiente comando en tu terminal:

pip install numpy pandas matplotlib seaborn xgboost scikit-learn tensorflow 

# Uso
1.Coloca el archivo del conjunto de datos (weatherAUS.csv) en la misma carpeta que el código fuente.
2.Abre una terminal y navega hasta la ubicación del archivo del código fuente.
3.Ejecuta el siguiente comando para ejecutar el código:

python nombre_del_archivo.py 

Asegúrate de reemplazar "nombre_del_archivo.py" con el nombre real del archivo de código fuente.
4.Observa la salida en la terminal. Verás las métricas de rendimiento para los modelos XGBoost y LSTM.
# Personalización
Puedes personalizar la ejecución del código utilizando argumentos de línea de comandos. Aquí tienes algunos ejemplos de cómo hacerlo:
•Cambiar el archivo del conjunto de datos:

python nombre_del_archivo.py --dataset ruta_del_archivo.csv 

Asegúrate de reemplazar "ruta_del_archivo.csv" con la ruta real del archivo del conjunto de datos.
•Cambiar la proporción de datos de prueba:

python nombre_del_archivo.py --test_size 0.2 
Puedes ajustar el valor después de "--test_size" para cambiar la proporción de datos de prueba.
•Cambiar la semilla aleatoria:

python nombre_del_archivo.py --random_state 42 

Puedes ajustar el valor después de "--random_state" para cambiar la semilla aleatoria.


