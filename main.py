
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import pandas as pd

app = Flask(__name__)

# CSV laden (Dummy)
try:
    df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv", sep=";", engine="python")
except Exception as e:
    df = None

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen beim KI-Telefonassistenten. Bitte stellen Sie Ihre Frage nach dem Piepton.", language="de-DE")
    response.record(
    max_length=10,
    action="https://telefon-qb6i.onrender.com/antwort",
    method="POST"
)
    return Response(str(response), mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    response = VoiceResponse()
    response.say("Vielen Dank für Ihre Nachricht. Wir melden uns in Kürze.", language="de-DE")
    return Response(str(response), mimetype="text/xml")

@app.route("/")
def home():
    return "KI-Telefonassistent läuft!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
