# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!
from json import *
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType,BotUttered
from swiplserver import PrologMQI
import json


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType


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
                    result = list(prolog_thread.query(f"{categoria}(X)."))

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

                if " " in objeto:
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
