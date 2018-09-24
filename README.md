# ML_News_Face_Recognition

### Trabajo de fin de master Big Data Analytics

#### - Definición del prototipo inicial

Este prototipo funcional tiene como finalidad detectar 
a personakes políticos, presentadores del telediario y 
la tipología de plano en los cuales estos personajes
 aparecen.
 
#### - Requerimientos generales del prototipo

* Modelo entrenado o método para detectar las caras de 
las personas que aparecen en cada frame del vídeo a 
analizar.

* Modelo entrenado o método para detectar rasgos 
carácterísticos de cada una de las caras identificadas.

* Modelo entrenado o método para obtener una representación
 universal de los rasgos identificados en cada cara, para 
 así tener una forma de comparar dos caras y saber si 
 pertenecen a la misma persona o no.
 
* APIs, librerías o software que nos permitan cargar e 
 interactuar con/modificar imágenes y vídeos. Además de
 poder cargar los modelos o métodos obtenidos para aplicarlos
 sobre las imágenes/frames de un vídeo.
 
* Datasets con imágenes con las caras de los personajes
políticos y presentadores de telediarios a análizar.

* Plataforma común donde poder hacer la integración de los
diferentes modelos y donde mostrar los resultados obtenidos.
Además se deberá tener en cuenta de que el prototipo soporte
la posibilidad de jugar y cambiar parámetros sin tener que 
cambiar el código original, para testear y obtener los mejores
resultados. Con el añadido de tener cuenta la posibilidad de
realizar el análisis en tiempo real.

#### - Evolución del prototipo

De los requerimientos mencionados a lunes 24 de Septiembre de
2018 se tienen todos cubiertos excepto el reconocimiento de 
planos. 

#### - Detalles de uso del prototipo

Todo el código se ha desarrollado siguiendo la versión 
*Python 3.6* obtenida desde la distribución *Anaconda*.

Las librerías usadas en el script que es necesario tener
instaladas son:

* cv2
* pickle
* numpy
* os
* sys
* dlib

##### - Modo de uso

###### - Entrenamiento

En primer lugar se han de generar los componentes más 
discriminantes de cada cara de las personas a reconocer,
 a través de los cuales posteriormente compararemos los
 componentes obtenidos en una imagen a análizar.
 
 Para generar este set de entrenamiento se han de introducir
 las imágenes, que se quieren usar como entrenamiento, en una
 carpeta dentro de la carpeta 'training' (contenida en la 
 carpeta 'images', por ejemplo, se pueden colocar en 
 cualquier otra carpeta). El nombre de la carpeta en la que se 
 introduzcan esas imágenes será la etiqueta que se usará
 para identificar al personaje que aparece en las imágenes.
 
 Para realizar el entrenamiento con esas imágenes se ha de
 ejecutar el siguiente script de la forma que sigue:
 
  ``` [Python]
  python generate_training_set.py 5_face_landmarks hog ./images/training
  ```
  
  Hay varias opciones disponibles de entrenamiento, la 
  descripción de cada una de estas opciones la he dejado 
  dentro del *script* en inglés, si se ejecuta:
  
  ``` [Python]
  python generate_training_set.py
  ```
  
  ###### - Reconocimiento
  
  Una vez entrenado el modelo con un método de reconocimiento
  de caras y un obtención de los rasgos carácterísticos 
  determinado, deberemos usar esos mismos métodos para el
  reconocimiento. Siguiendo con el ejemplo anterior, usaremos 
  el método '5_face_landmarks' junto con el
  detector 'hog'. Veamos un ejemplo aplicado a una imagen:
  
   ``` [Python]
    python face_recognition.py 5_face_landmarks hog ./images/test/pedro.jpg 10 yes 0.475
   ```
   
   Y para un vídeo la forma de llamar al script para su
   análisis es análoga:
   
   ``` [Python]
    python face_recognition.py 5_face_landmarks hog ./video/test/pedro.jpg 10 yes 0.475
   ```
   
   De la misma forma que antes, si queremos ver todas las
   opciones podemos introducir lo siguiente:
   
   ``` [Python]
   python face_recognition.py
   ```