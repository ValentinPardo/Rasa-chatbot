# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!
from json import *
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType,BotUttered

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    @staticmethod
    def fetch_slots(tracker: Tracker) -> List[EventType]:
        #Collect slots that contain the user's name and phone number.

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

        dispatcher.utter_message("utter_saludar")

        events.append(dispatcher.utter_message("utter_saludar"))


        # any slots that should be carried over should come after the
        # `session_started` event
        events.extend(self.fetch_slots(tracker))
        
        # an `action_listen` should be added at the end as a user message follows
        msg="hola como estas?"
        events.append(dispatcher.utter_message(text=msg))

        events.append(ActionExecuted("action_listen"))

        return events

#ManipularJson
"""class OperarArchivo():
    @staticmethod
    def guardar(AGuardar):
        with open() as archivo_descarga:
            json.dump(AGuardar, archivo_descarga, indent=4)
        archivo_descarga.close()

    @staticmethod
    def cargarArchivo():
        if os.path.isfile():
"""

objt_db = {
    'bicicleta': 'costo: su valor es $1000',
    'bateria': 'Costo: $60500',
    'parlante': 'Costo: $7500',
    'pelota de futbol': 'Costo: $4500',
    'celular barato': 'Costo: $23000',
    'celular caro': 'Costo: $120000',
    'celular medio': 'Costo: $40000',
    'zapatillas': 'Costo: $25000'
}

class AccionInformacionCompra(Action):
    def name(self) -> Text:
        return "accion_informar_compra"
    
    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:
        objeto_actual = next(tracker.get_latest_entity_values("objeto"),None)


        objt = objt_db.get(objeto_actual,None)
        if not objt:
            msg = f"No he encontrado ninguna opcion disponible sobre {objeto_actual} en este momento :( "
            dispatcher.utter_message(text=msg)
            return[]
        
        msg = f"Creo que esto es lo que estas buscando: \n {objt_db[objeto_actual]} "
        dispatcher.utter_message(text=msg)

        #SlotSet("compra",objeto_actual)
        return[]
        
        