
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from openai import OpenAI
import os
import pandas as pd

app = Flask(__name__)
client = OpenAI(api_key="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")

# Lade Kundendaten CSV in DataFrame
df = pd.read_csv("verknuepfte_tabelle_final_bereinigt.csv", sep=';', engine='python')

@app.route("/voice", methods=["POST"])
def voice():
    response = VoiceResponse()
    user_input = request.form.get('SpeechResult', '')
    caller_number = request.form.get('From', '')

    # Suche in der Datenbank nach Telefonnummer oder Bestellnummer im User Input
    customer_info = df[df['RA Tel'].astype(str).str.contains(caller_number[-7:], na=False)]
    for idx, row in customer_info.iterrows():
        bestellnummer = row.get('Bestellnummer', '')
        produkt = row.get('product_name', '')
        status = row.get('order_status', '')
        tracking = row.get('shipping_tracking', '')
        break
    else:
        bestellnummer = produkt = status = tracking = 'nicht gefunden'

    # GPT vorbereiten mit Kundendaten
    system_prompt = f"Du bist ein deutscher E-Commerce Kundensupport Bot. Die Bestellnummer ist {bestellnummer}, Produkt: {produkt}, Status: {status}, Trackingnummer: {tracking}. Antworte freundlich, kurz und hilfsbereit."

    completion = client.chat.completions.create(
        model="gpt-4o-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    gpt_response = completion.choices[0].message.content
    response.say(gpt_response, language='de-DE', voice='Polly.Vicki')
    response.gather(input='speech', language='de-DE', timeout=5)

    return Response(str(response), mimetype='text/xml')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
