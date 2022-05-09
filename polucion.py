# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import utils
# %matplotlib inline
pd.set_option('display.max_colwidth', None)

df_pol = pd.read_csv("/content/data.csv", sep=",")
del df_pol["No"]

# Parte 1: Preprocesamiento de los datos:

print("Número de características: {}".format(df_pol.shape[1]))
print("Número de muestras {}\n".format(df_pol.shape[0]))

print("Datos nulos por característica:")
df_pol.isnull().sum()

df_pol.head(100000)

"""Analicemos ahora las estadísticas de nuestro dataframe:"""

df_pol.describe()

"""## Eliminando datos nulos:
Analizando el dataset encontramos que la característica principal (PM 2.5) era la única que contenía datos nulos, siendo esta nuestra variable a predecir, decidimos que lo mejor para no sesgar nuestro algoritmo era eliminar estos datos nulos y no realizar ninguna aproximación.
"""

df_pol = df_pol.dropna()
print("Número de características: {}".format(df_pol.shape[1]))
print("Número de muestras {}\n".format(df_pol.shape[0]))

"""Comprobamos ahora que nuestro dataset esta libre de datos nulos"""

df_pol.isna().any()

"""## Datos categóricos:
Para los datos categóricos se llegó a la conclusión que en términos de polución la dirección del viento es fundamental pero el algoritmo de entrenamiento no entenderá estos datos de forma categórica, por eso se procedió a codificar estos datos, se ve que los  datos Nominales, esto quiere decir que no tienen una jerarquía, sería problemático para nuestro algoritmo asignarle valores (1,2,3,4) a cada dirección del viento, así es preferible usar variables *Dummie* y tratar estos datos como binarios, de este modo se obtendrán predicciones más precisas al ejecutar el algoritmo de machine learning.
Al crear estas variables e insertarlas en nuestra tabla, se eliminará la variable cbwd, e ingresaran 3 columnas ¿Por qué 3? se puede eliminar una de ellas y asumir que esta será la ausencia de las otras, así no se saturara la tabla con muchas columnas nuevas.

Para esto se usara  ``` pd.get_dummies ``` y se le dara  ``` drop_first = True ``` para eliminar la primera columna, es decir, la variable NE significará que las demás están en cero.

### Dirección combinada del viento (cbwd)
"""

print("Primero analizamos qué categorías tenemos: ")
df_pol.cbwd.value_counts()

"""**CV** no es un punto cardinal, investigando obtuvimos que cv es un valor **errado** en traducción, y que corresponde a la dirección SW, vamos a corregirlo rápidamente:"""

df_pol['cbwd'].replace('cv', 'SW', inplace = True)
df_pol.cbwd.value_counts()

dummies = pd.get_dummies(df_pol['cbwd'], drop_first=True)
dummies.head(40000)

"""Hecho esto, procedemos a descartar la columna cbdw de nuestro dataframe y le daremos por otro lado las columnas NW, SE y SW que serán binarias."""

df_pol = df_pol.drop(['cbwd'], axis=1)
df_pol = pd.concat([df_pol, dummies], axis=1)

print("Demos un vistazo ahora a nuestro dataframe: ")
df_pol.head(10)

"""## Estadísticas de cada característica:
Obtendremos la cuenta, media, mínimos, máximos y otra información relevante sobre nuestros datos, nos será muy útil para determinar la naturaleza de nuestros datos. Por ejemplo, vemos los quartiles (min, 25%, 50%, 75%, max). 

Si analizamos esto, podremos ver que los máximos pueden alejarse abruptamente del 75%. Esto nos puede dar una pista de que los datos con valores tan altos pueden deberse a errores de medición por lo que podemos despreciarlos, dado que no son valores reales y pueden tender a sesgar el algoritmo si hacemos algo con ellos.
"""

df_pol.describe()

# Gráficos por variable

df_pol.boxplot(['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir'])

df_pol.drop(['Is', 'Ir'], axis=1, inplace=True)
limpiar = ['pm2.5','Iws']
for i in limpiar:
    per25 = df_pol[i].quantile(0.25)
    per75 = df_pol[i].quantile(0.75)
    IQR = per75 - per25
    UpperLimit = per75 + 1.5*IQR
    LowerLimit = per25 - 1.5*IQR
    df_pol = df_pol.loc[(df_pol[i] < UpperLimit) & (df_pol[i] > LowerLimit)]

print("Numero de muestras despues de limpieza: {}".format(df_pol.shape[0]))

df_pol.boxplot(['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws'])

df_pol.describe()

df_pol_norm = df_pol.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_pol_norm.head(40000)

"""Podremos ver ahora en nuestro diagrama de cajas y bigotes, todos nuestros datos expresados en las mismas unidades. Por lo que al juntar nuestros gráficos tendremos un vistazo más simple y agradable para poder dar un análisis más correcto."""

df_pol_norm.boxplot(['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws'])

# Analizando correlaciones:

corr = df_pol.corr()
corr.style.background_gradient(cmap='coolwarm' , axis=None)

"""## Graficos de dispersion 

Esto es otra representancion de correlaciones de la caracteristicas. Entre más juntos estén los puntos más relacionadas estarán las variables. Observamos que, al igual que en el cuadro de correlaciones, la diagonal pricipal tiene una correlación de 1 o los datos forman gráficos de barras dado que están directamente correlacionados, estos no los tenemos en cuenta por que es la relación entre una variable consigo misma.
"""

sns.set(style="whitegrid", context="notebook")
sns.pairplot(df_pol[["pm2.5","DEWP","TEMP","PRES","Iws"]])

sns.set(style="whitegrid", context="notebook")
corr = df_pol.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
ax = sns.heatmap(corr, annot=True, fmt=".1%", linewidths=.5, mask = mask, vmax=1, vmin = 0)


df_pol.boxplot(['pm2.5', 'TEMP', 'Iws', 'PRES', 'DEWP'])

first_column = df_pol.pop('pm2.5')
df_pol.insert(0, 'pm2.5', first_column)

df_pol.head(40000)

df_pol.to_csv('dataset_limpio.csv', index=False)

"""## Conclusiones

Después de analizado el dataset, encontramos que habían algunas características altamente correlacionadas aun así se deben dejar pues son necesarias para el algoritmo que se implementara despues. Hecho esto, se procedio a normalizar los datos, puesto que estaban en unidades diferentes y esto es bastante perjudicial al momento de entrenar el algoritmo. Como punto final, en el último diagrama de cajas y bigotes, ciertas características parecen tener datos atípicos, pero analizándolos a fondo consideramos que son datos normales y que no se trata de errores de medición, así pues, no fueron eliminados puesto que son fundamentales a la hora de entrenar adecuadamente nuestro algoritmo.

# Parte 2: Entrenando nuestro algoritmo
Ya optimizamos nuestro dataset para ser entrentado, posteriormente lo que haremos es separar nuestros datos en datos de entrenamineto y datos de prueba, nuestro porcentaje inicial para separarlos será 80%, 20% pero esto no será definitivo, será a partir de las pruebas que definiremos qué valores nos entregarán predicciones más confiables.
"""

# Separar los datos en entrenamiento y prueba
X=df_pol.iloc[:,1:].values
y=df_pol["pm2.5"].values



# Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalizar los datos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Aplicar algoritmos de regresion para etiquetas continuas
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
# Medir la precisión de los algoritmos
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

"""### Metodología 
Se usarán 3 diferentes tipos de algoritmos de regresión para ver cuál de estos se acomodan mejor a los datos que se tienen y dan la predicción optima, Random Forest Regression, Bagging Regression, Extra Trees Regression, dicho esto se procede a entrenar cada uno de los algoritmos y a sacar las métricas de eficacia de estos para su posterior análisis
"""

# Random Forest Regression
random_reg = RandomForestRegressor()
random_reg.fit(X_train, y_train)
y_pred = random_reg.predict(X_test)
print("Random Forest Regression:")
print("Score: {}".format(random_reg.score(X_test, y_test)))

# Mean Squared Error
print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_pred)))

# Root Mean Squared Error
print("Root Mean Squared Error: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))


# MAD (Median Absolute Deviation)
print("MAD: {}".format(mean_absolute_error(y_test, y_pred)))

# MAPE (Mean Absolute Percentage Error)
print("MAPE: {}".format((mean_absolute_percentage_error(y_test, y_pred)) * 100))

print("\n")

# Grafico de precisión
plt.figure(figsize=(10, 6))
plt.plot(y_test, y_pred, 'o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# Bagging Regression
bag_reg = BaggingRegressor()
bag_reg.fit(X_train, y_train)
y_pred = bag_reg.predict(X_test)
print("Bagging Regression:")
print("Score: {}".format(bag_reg.score(X_test, y_test)))

# Mean Squared Error
print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_pred)))

# Root Mean Squared Error
print("Root Mean Squared Error: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

# MAD (Median Absolute Deviation)
print("MAD: {}".format(mean_absolute_error(y_test, y_pred)))

# MAPE (Mean Absolute Percentage Error)
print("MAPE: {}".format((mean_absolute_percentage_error(y_test, y_pred)) * 100))


print("\n")

# Grafico de precisión
plt.figure(figsize=(10, 6))
plt.plot(y_test, y_pred, 'o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# Extra Trees Regression
extra_reg = ExtraTreesRegressor()
extra_reg.fit(X_train, y_train)
y_pred = extra_reg.predict(X_test)
print("Extra Trees Regression:")
print("Score: {}".format(extra_reg.score(X_test, y_test)))

# Mean Squared Error
print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_pred)))

# Root Mean Squared Error
print("Root Mean Squared Error: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

# MAD (Median Absolute Deviation)
print("MAD: {}".format(mean_absolute_error(y_test, y_pred)))

# MAPE (Mean Absolute Percentage Error)
print("MAPE: {}".format((mean_absolute_percentage_error(y_test, y_pred)) * 100))


print("\n")

# Grafico de precisión
plt.figure(figsize=(10, 6))
plt.plot(y_test, y_pred, 'o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

"""## Realizando algunas predicciones"""

# Predecir el valor de una nueva observación
new_observation = np.array([[2015, 1, 1, 0, -16, -4, 1020, 1.79, 0, 1, 0]])
new_observation = sc.transform(new_observation)
new_prediction = extra_reg.predict(new_observation)
print("Prediccion 2015/1/1 con valores de inicios de 2010: {}".format(new_prediction))

# Predecir el valor de una nueva observación
new_observation = np.array([[2015, 1, 2, 0, -16, -4, 1020, 1.79, 0, 1, 0]])
new_observation = sc.transform(new_observation)
new_prediction = extra_reg.predict(new_observation)
print("Prediccion 2015/1/2 con valores de inicios de 2010: {}".format(new_prediction))

# Predecir el valor de una nueva observación
new_observation = np.array([[2015, 1, 3, 0, -19, -1, 1027, 51.84, 1, 0, 0]])
new_observation = sc.transform(new_observation)
new_prediction = extra_reg.predict(new_observation)
print("Prediccion 2015/1/3 con valores de finales de 2014: {}".format(new_prediction))

# Predecir el valor de una nueva observación
new_observation = np.array([[2015, 1, 4, 0, -19, -1, 1027, 51.84, 1, 0, 0]])
new_observation = sc.transform(new_observation)
new_prediction = extra_reg.predict(new_observation)
print("Prediccion 2015/1/4 con valores de finales de 2014: {}".format(new_prediction))

"""Para inicios de 2010 los valores de polucion eran demasiado altos mientras que para 2014 ya estaban tendiendo a bajar, a juzgar por esto, los datos predichos tienen bastante sentido, aunque no podemos estar 100% seguros de esto dado que no contamos con los resultados reales ya para estas fechas.