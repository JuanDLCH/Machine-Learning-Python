# Machine-Learning-Python
Preprocesamiento de un dataset y entrenamiento de un algoritmo de ML

Todos la la información está incluida en el archivo polucion.ipynb

# Entrenando un algoritmo de Machine Learning para predecir la contaminación en Beijing

## Integrantes:
- Juan Diego Londoño
- Mario Alejandro Saldarriaga

## FORMULACIÓN DEL PROBLEMA
1. ¿El aprendizaje automático es apropiado para este problema y por qué si o no?
    - Si es apropiado para este problema pues lo que se pretende hacer es entrenar a un algoritmo para que sea capaz de predecir la cantidad de contaminación que puede haber en un determinado sitio en una determinada fecha.

2. ¿Cuál es el problema de ML si hay uno y cómo sería la métrica de éxito?
    - EL problema de ML es el poder entrenarlo para predecir, esto se hará con regresión, la métrica de éxito seria lograr llegar a uno de los valores que se tienen para la característica pm2.5.

3. ¿Qué tipo de problema de ML es este?
    - Es un problema de predicción mediante el método de regresión
    
4. ¿Los datos son apropiados? 
    - Si, pues se cuenta con no solo datos de la contaminación, sino también del comportamiento del clima, lo cual es una parte influyente en la distribución de esa contaminación en Beijing



## [Dataset](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data):
Tenemos un dataset con **43824 Muestras** referentes a las partículas por millón (PM 2.5) presentes en el aire de Beijing, China por hora entre los años 2010 y 2014 y con **12 Características**:

### Características:
- No: Fila
- Año
- Mes
- Dia
- Hora
- PM 2.5: Concentración de partículas por millón (PM2.5), medida en $ug/m^3$
- DEWP: Punto de rocío, medido en °F
- TEMP: Temperatura media en °F
- PRES: Preción atmosférica, medida en hPa (unidades de presión)
- cbwd: Dirección del viendo combinada
- Iws: Velocidad del viento acumulada medida en m/s
- Is: Horas de nieve acumuladas
- Ir: Horas de lluvia acumuladas


# Librerías a utilizar:
- Pandas: ``` pip install pandas ```
- Numpy: ``` pip install numpy ```
- Seaborn: ``` pip install seaborn ```
- Matplotlib Pyplot: ``` pip install matplotlib ```

# Importamos estas librerías y archivos a nuestro código:
- Configuraremos las tablas de pandas quitando la limitación del ancho de columna
- Importamos el dataset, llamado data.csv y lo asignaremos a una variable llamada df_pol
- Eliminaremos la columna que nos indica el número de fila, aquí no la necesitamos


# Hecho por:
<table>
  <tr>
    <td align="center"><a href="https://github.com/JuanDLCH"><img src="https://avatars.githubusercontent.com/u/53449798?v=4" width="100px;" alt=""/><br /><sub><b>Juan Diego Londoño Ch</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/Alejandro-96"><img src="https://avatars.githubusercontent.com/u/65933953?v=4" width="100px;" alt=""/><br /><sub><b>Alejandro Saldarriaga Alv</b></sub></a><br /></td>
  </tr>
</table>
  
