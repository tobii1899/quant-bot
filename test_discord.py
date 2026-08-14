import os
from datetime import datetime
import requests

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1537949378126286879/k1RsdmZjARU3ZlGcdGe_WwgdOMkCXHV7dd1Nz_v3j1qyZ2VetKshjzC516dSUeBVa1vi"


def main():
    now_str = datetime.now().strftime("%d.%m.%Y um %H:%M:%S")

    payload = {
        "username": "AAPL Trading Bot (GitHub Actions)",
        "content": (
            f"🧪 **Test Versuch via GitHub Actions**\n"
            f"Der Bot-Workflow auf den GitHub-Servern läuft sauber durch!\n"
            f"📅 *Ausgeführt am {now_str} CEST*"
        ),
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code in [200, 204]:
            print("✅ Test-Nachricht erfolgreich via GitHub Actions gesendet!")
        else:
            print(f"⚠️ Fehler: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Fehler: {e}")


if __name__ == "__main__":
    main()