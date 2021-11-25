##Importacion de libreria
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#Lectura de dataset de los vinos
df= pd.read_csv("wine-clustering.csv")

##Preprocesamiento de datos
X=df.values

#Estandarizacion de caracteristicas
sc= StandardScaler()
sc.fit(X)
X=sc.transform(X)
#Reduccion de dimensionalidad hasta obtener una varianza de 90% o mas
pca=PCA(n_components=8)
pca.fit(X)
X_PCA=pca.transform(X)
Var =pca.explained_variance_ratio_
print('Varianza:',sum(Var))

#Declaracion de variables auxiliares
Kinicial=2
Kfinal=20
step=1
Silueta=[]
K=[]
Silueta_PCA=[]
K_PCA=[]

##Comprobacion de mejor k y random state para el conjunto de datos estudiado, sin PCA y con PCA

#Se prueban diferentes valores para random state
for rs in range(0,20):
  #Limpiado de listas auxiliares
  Silueta_aux=[]
  K_aux=[]

  Silueta_aux_PCA = []
  K_aux_PCA = []

  #Por cada random state, se prueban diferentes k
  for k in range (Kinicial,Kfinal,step):
    #Se inicia con la evaluacion de k-means
    kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                    n_clusters=k, n_init=20,random_state=rs, tol=0.0001, verbose=0).fit(X)

    kmeans_PCA = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                    n_clusters=k, n_init=20, random_state=rs, tol=0.0001, verbose=0).fit(X_PCA)

    #Se obtienen los resultados de la prediccion de k-means
    labels = kmeans.labels_
    labels_PCA = kmeans_PCA.labels_
    #Se guarda el puntaje de silueta calculado con la metrica euclidiana y el k con el que se obtuvo
    Silueta_aux.append(silhouette_score(X,labels,metric='euclidean'))
    K_aux.append(k)
    Silueta_aux_PCA.append(silhouette_score(X_PCA, labels_PCA, metric='euclidean'))
    K_aux_PCA.append(k)
  #Se guardan los puntajes de siluetas y k de cada de una de las comprobaciones en una lista de listas
  Silueta.append(Silueta_aux)
  K.append(K_aux)

  Silueta_PCA.append(Silueta_aux_PCA)
  K_PCA.append(K_aux_PCA)


#Variables auxiliares
max_value=[]
k_list=[]
max_value_PCA=[]
k_list_PCA=[]

#Se obtienen los puntajes de silueta mas alto y los k con los que se obtuvieron
for i in range(0,len(Silueta)):
    max_value.append(max(Silueta[i]))
    k_list.append(K[i][Silueta[i].index(max_value[i])])

    max_value_PCA.append(max(Silueta_PCA[i]))
    k_list_PCA.append(K_PCA[i][Silueta_PCA[i].index(max_value_PCA[i])])

#Se obtiene el valor maximo de los puntajes de silueta y
max_score= max(max_value)
max_score_PCA= max(max_value_PCA)

#Se obtiene en que posicion estaba el puntaje de silueta maximo, para asi, saber con que valor
#de random state se obtuvo
state= max_value.index(max_score)
state_PCA= max_value_PCA.index(max_score_PCA)

#Se obtienen el k con el que se obtuvo el puntaje de silueta maximo
k_elegido= k_list[state]
k_elegido_PCA= k_list_PCA[state_PCA]

print("El mejor coeficiente de silueta sin PCA dio igual a:",max_score,",esto se obtuvo con el parametro k igual a:",
      k_elegido,"y con el parametro random state igual a:",state)


print("El mejor coeficiente de silueta con PCA dio igual a:",max_score_PCA,",esto se obtuvo con el parametro k igual a:",
      k_elegido_PCA,"y con el parametro random state igual a:",state_PCA)





#Se grafican los puntajes de silueta para los diferentes K y random states, para los datos con PCA y sin PCA

plt.figure(1)
plt.title("(Grafica de datos sin PCA) K vs puntaje de silueta", fontdict=None, loc='center', pad=None)
plt.xlabel('K', fontsize=16)
plt.ylabel('Puntaje de coeficiente de silueta', fontsize=16)
plt.plot(K, Silueta,marker='o')

plt.figure(2)
plt.title("(Grafica datos con PCA) K vs puntaje de silueta", fontdict=None, loc='center', pad=None)
plt.xlabel('K', fontsize=16)
plt.ylabel('Puntaje de coeficiente de silueta', fontsize=16)
plt.plot(K, Silueta_PCA,marker='o')




#Obtencion de centroides y agrupaciones (con y sin PCA)
kmeans1 = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                    n_clusters=k_elegido, n_init=20,random_state=state, tol=0.0001, verbose=0).fit(X)
labels1 = kmeans1.labels_

kmeans1_pca = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                    n_clusters=k_elegido_PCA, n_init=20,random_state=state_PCA, tol=0.0001, verbose=0).fit(X_PCA)
labels1_pca = kmeans1_pca.labels_

centroides=kmeans1.cluster_centers_
centroides_PCA= kmeans1_pca.cluster_centers_


#Se le asignan colores especificos a los datos de cada agrupacion que fue obtenida por K-means
colores=['red','green','blue']
color_datos=[]
for ii in labels1:
    color_datos.append(colores[ii])

color_datos_PCA=[]
for ii in labels1_pca:
    color_datos_PCA.append(colores[ii])

#Grafica de centroides sin PCA
plt.figure(3)
plt.title("Grafica de centroides sin PCA", fontdict=None, loc='center', pad=None)
plt.scatter(X[:,0],X[:,1],c=color_datos,alpha=0.5)
plt.scatter(centroides[:, 0], centroides[:, 1], marker='X', c=colores, s=300)


#Grafica de centroides con PCA
plt.figure(4)
plt.title("Grafica de centroides con PCA", fontdict=None, loc='center', pad=None)
plt.scatter(X_PCA[:,0],X_PCA[:,1],c=color_datos_PCA,alpha=0.5)
plt.scatter(centroides_PCA[:, 0], centroides_PCA[:, 1], marker='X', c=colores, s=300)
plt.show()