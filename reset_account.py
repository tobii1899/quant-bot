import os
import requests
from dotenv import load_dotenv

load_dotenv("keys.env")

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

# Alpaca Paper API V2 Base URL
base_url = "https://paper-api.alpaca.markets/v2"

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
}

# 1. Aktuellen Cash-Stand abfragen
r = requests.get(f"{base_url}/account", headers=headers)
cash = float(r.json()["cash"])
print(f"Aktuelles Guthaben: ${cash:,.2f}")

# 2. Differenz zu $6.000 berechnen
target = 6000.0
delta = target - cash

if delta == 0:
    print("Konto steht bereits exakt auf $6.000!")
else:
    # Positive Delta = Einzahlung (IN), Negative Delta = Auszahlung (OUT)
    direction = "IN" if delta > 0 else "OUT"
    amount = abs(delta)
    
    payload = {
        "entry_type": direction,
        "amount": str(amount)
    }
    
    # Sendet eine Korrektur-Buchung an den Paper-Account
    res = requests.post(f"{base_url}/account/journal", json=payload, headers=headers)
    
    if res.status_code == 200 or res.status_code == 201:
        print(f"✅ Erfolgreich angepasst! Neuer Kontostand: ${target:,.2f}")
    else:
        print(f"❌ Fehler bei der Buchung: {res.text}")