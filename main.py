
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import pandas as pd

app = Flask(__name__)
df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv", sep=';', engine='python')

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen beim KI Telefonassistenten. Bitte stellen Sie Ihre Frage nach dem Piepton.", language="de-DE")
    response.record(max_length=10, action="/antwort", method="POST")
    return Response(str(response), mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    response = VoiceResponse()
    response.say("Vielen Dank für Ihre Nachricht. Wir melden uns in Kürze.", language="de-DE")
    return Response(str(response), mimetype="text/xml")
