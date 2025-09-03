
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import pandas as pd

app = Flask(__name__)

# CSV-Datei laden (bei Bedarf auf echten Pfad ändern)
df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv", sep=";", engine="python")

@app.route("/telefon", methods=["POST"])
def telefon():
    response = VoiceResponse()
    response.say("Willkommen beim KI-Telefonassistenten. Bitte stellen Sie Ihre Frage nach dem Piepton.", language="de-DE")
    response.record(
        max_length=10,
        action="https://telefon-qb6i.onrender.com/antwort",  # absolute URL für Twilio
        method="POST"
    )
    return Response(str(response), mimetype="text/xml")

@app.route("/antwort", methods=["POST"])
def antwort():
    # Für die Demo: einfache Antwort zurückgeben
    response = VoiceResponse()
    response.say("Vielen Dank. Ihre Frage wird bearbeitet. Auf Wiederhören!", language="de-DE")
    return Response(str(response), mimetype="text/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
