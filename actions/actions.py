# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!

from typing import Any, Text, Dict, List

#JSON
import json
#RASA
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType

#PROLOG
from swiplserver import PrologMQI


#CLASIFICACION DATOS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#ARBOL DE DECISION
from sklearn.tree import DecisionTreeClassifier,plot_tree

#REDES NEURONALES
from keras import Sequential,layers
from keras.layers import Dense,Input
from keras.utils import to_categorical
import numpy as np

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    @staticmethod
    def fetch_slots(tracker: Tracker) -> List[EventType]:
        """Collect slots that contain the user's name and phone number."""

        slots = []
        for key in ("name", "phone_number"):
            value = tracker.get_slot(key)
            if value is not None:
                slots.append(SlotSet(key=key, value=value))
        return slots

    async def run(
      self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        # the session should begin with a `session_started` event
        events = [SessionStarted()]

        # any slots that should be carried over should come after the
        # `session_started` event
        events.extend(self.fetch_slots(tracker))

        # an `action_listen` should be added at the end as a user message follows
        events.append(ActionExecuted("action_listen"))

        return events

#ManipularJson


class AccionOpciones(Action):
    def name(self) -> Text:
        return "accion_opciones"
    async def run(
            self, dispatcher:CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:
        categoria = next(tracker.get_latest_entity_values("categoria"),None)

        with PrologMQI(port=8000) as mqi:
            with mqi.create_thread() as prolog_thread:
                prolog_thread.query("consult('C:/Users/elmin/OneDrive/Escritorio/Uni y Prog/Prog. Exploratoria/data/conocimiento.pl')")

                result = prolog_thread.query(f"{categoria}(_).")
                
                if not result:
                    dispatcher.utter_message(text=f"Lo siento, no encontre nada relacionado a la categoria {categoria}")
                    return[]
                else:
                    result = list(prolog_thread.query(f"{categoria}(X).")) # type: ignore

                dispatcher.utter_message(text=f"Estos son los {categoria}s disponibles: \n" + str(result))

        SlotSet("categoria",categoria)
        return[]

class AccionInformacionCompra(Action):
    def name(self) -> Text:
        return "accion_informar_compra"
    
    async def run(
            self, dispatcher:CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:
        categoria = tracker.get_slot("categoria")

        objeto = next(tracker.get_latest_entity_values("objeto"),None)

        with PrologMQI(port=8000) as mqi:
            with mqi.create_thread() as prolog_thread:

                prolog_thread.query("consult('C:/Users/elmin/OneDrive/Escritorio/Uni y Prog/Prog. Exploratoria/data/conocimiento.pl')")
                
                print("llego")

                if " " in objeto: # type: ignore
                    result = prolog_thread.query(f"{categoria}('{objeto}').")
                else:
                    result = prolog_thread.query(f"{categoria}({objeto}).")
                
                if not result:
                    dispatcher.utter_message(text=f"Lo siento, no encontre nada relacionado a {objeto}")
                    return[]
                else:
                    # Cargar el archivo JSON en una variable de Python
                    with open("C:/Users/elmin/OneDrive/Escritorio/Uni y Prog/Prog. Exploratoria/data/info.json", "r") as archivo:
                        datos_cargados = json.load(archivo)

                    # Buscar los datos por una key específica
                    if not result:
                        precio = datos_cargados[f"'{objeto}'"]["precio"]
                        descripcion = datos_cargados[f"'{objeto}'"]["descripcion"]
                    else:
                        precio = datos_cargados[f"{objeto}"]["precio"]
                        descripcion = datos_cargados[f"{objeto}"]["descripcion"]

                dispatcher.utter_message(text=f"Tal vez esto te interese :) \n Precio: {str(precio)} \n Descripcion: {str(descripcion)} ")
        
        SlotSet("objeto",objeto)
        return[]

class ActionSaludar(Action):
    def name(self):
        return "accion_saludar"

    def run(self, dispatcher, tracker, domain):
        nombre = tracker.get_slot("nombre")
        if nombre:
            mensaje = f"Hola, {nombre}! ¿como estas?"
        else:
            mensaje = "Hola! ¿como estas?"

        dispatcher.utter_message(mensaje)
        return []

class RedNeuronal(Action):
    def name(self):
        return "accion_tal_vez_te_intereseX"
    
    def run(self, dispatcher, tracker, domain):
        mensaje = " tal vez te interese un objeto, es asi?"
        dispatcher.utter_message(mensaje)
        #test de red neuronal con slots edad,sexo,estudios
        return []

#preprocesado de datos
df = pd.read_csv('C:/Users/elmin/OneDrive/Escritorio/Uni y Prog/Prog. Exploratoria/data/treeData.csv', engine='python', index_col=0)
pf = pd.read_csv('C:/Users/elmin/OneDrive/Escritorio/Uni y Prog/Prog. Exploratoria/data/treeData.csv', engine='python', index_col=0)
label_encoder = LabelEncoder()

#arbol decision (en teoria funcionando)
#df['categoria_objetivo'] = label_encoder.fit_transform(df['categoria_objetivo'])
#df['sexo'] = label_encoder.fit_transform(df['sexo'])
#df['estudios'] = label_encoder.fit_transform(df['estudios'])
#explicativas = df.drop(columns='categoria_objetivo')
#objetivo = df.categoria_objetivo
#model = DecisionTreeClassifier(max_depth=4)
#model.fit(X=explicativas,y=objetivo)
#plot_tree(decision_tree=model, feature_names=explicativas.columns, filled=True); # type: ignore

#red neuronal
pf['categoria_objetivo'] = label_encoder.fit_transform(pf['categoria_objetivo'])
pf['sexo'] = label_encoder.fit_transform(pf['sexo'])
pf['estudios'] = label_encoder.fit_transform(pf['estudios'])
#division del dataset
X = pf[['sexo','edad','estudios']].values
y = pf['categoria_objetivo'].values

y = to_categorical(y, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#MODELO
model = Sequential([
    layers.Input(shape=(X_train.shape[1])),
    layers.Dense(units=64, activation='relu'),
    #layers.Dropout(0.3),  # Ejemplo de capa Dropout
    layers.Dense(units=16, activation='relu'),
    layers.Dense(units=4, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#ENTRENAMIENTO
model.fit(X_train,y_train, epochs=5, batch_size=25, validation_data=(X_test,y_test))
loss, accuracy = model.evaluate(X_test,y_test)
print(f'loss: {loss}, Presicion: {accuracy}')

#PRUEBA DE LA RED
# Supongamos que tienes una lista de categorías para 'sexo' y 'estudios'
categorias_sexo = ['hombre', 'mujer']
categorias_estudios = ['primaria', 'secundaria', 'universitarios']

# Crear el objeto LabelEncoder para 'sexo' y 'estudios'
label_encoder_sexo = LabelEncoder()
label_encoder_estudios = LabelEncoder()

# Ajustar y transformar las categorías
label_encoder_sexo.fit(categorias_sexo)
label_encoder_estudios.fit(categorias_estudios)

# Definir valores para 'sexo' y 'estudios'
sexo = 'hombre'
edad = 50
estudios = 'universitarios'

# Transformar las categorías en números
categoria_sexo_transformada = label_encoder_sexo.transform([sexo])[0] # type: ignore
categoria_estudios_transformada = label_encoder_estudios.transform([estudios])[0] # type: ignore

# Crear el array NumPy con las variables definidas
sample_categoria = np.array([[categoria_sexo_transformada, edad, categoria_estudios_transformada]])



prediction = model.predict(sample_categoria)
print(f"Precision de la prediccion: {prediction}")

predicted_class = np.argmax(prediction, axis=1)
print(f"Clase predicha: {predicted_class}")

print(pf.sample(10))
print("......................")

clase_predicha = np.argmax(prediction, axis=1)
print(f"datos input: {sample_categoria}")
print(f"Salida: {clase_predicha}")

print("----------------------------------SEGUNDO EJEMPLO----------------------------------")

# Crear el array NumPy con las variables definidas
sample_categoria = np.array([[0,19,0]])



prediction = model.predict(sample_categoria)
print(f"Precision de la prediccion: {prediction}")

predicted_class = np.argmax(prediction, axis=1)
print(f"Clase predicha: {predicted_class}")

print(pf.sample(10))
print("......................")

clase_predicha = np.argmax(prediction, axis=1)
print(f"datos input: {sample_categoria}")
print(f"Salida: {clase_predicha}")

