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
from pyswip import Prolog
from swiplserver import PrologMQI


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
        # any slots that should be carried over should come after the
        # `session_started` event
        events.extend(self.fetch_slots(tracker))
        
        # an `action_listen` should be added at the end as a user message follows

        events.append(ActionExecuted("action_listen"))

        return events

#ManipularJson


class AccionInformacionCompra(Action):
    def name(self) -> Text:
        return "accion_informar_compra"
    
    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:
        objeto_actual = next(tracker.get_latest_entity_values("objeto"),None)

        with PrologMQI(port=8000) as mqi:
            with mqi.create_thread() as prolog_thread:
                prolog_thread.query(r"consult('C:\Users\elmin\OneDrive\Escritorio\Uni y Prog\Prog. Exploratoria\data\conocimiento.pl')")
                response = prolog_thread.query(f"objeto({objeto_actual},X,Y)")

                print(str(response))
                dispatcher.utter_message(text=f"Respuesta: {str(response)}")

        #SlotSet("compra",objeto_actual)
        return[]
        
        