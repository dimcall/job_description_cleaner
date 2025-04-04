import requests
# URL of the server
url = "http://localhost:8000/predict"
# Data to be sent to the server
data = [
    {
        "html": """
                A livello tecnico gestisce inoltre i responsabili Finanze e contabilit delle unit aziendali e di gestione.      Il suo profilo
                - Vanta una pluriennale esperienza in ambito dirigenziale e dispone di competenze nell'ambito finanziario, preferibilmente acquisite in una grande azienda
                - Ha conseguito un diploma universitario o di formazione equivalente e possiede provate nozioni approfondite di economia aziendale
        """
    }
]

# Response format:
response = requests.post(url, json=data)

if response.status_code == 200:
    for i, item in enumerate(response.json()):
        print(f"\n Cleaned Job Description #{i+1}:\n")
        print(item["cleaned"])
else:
    print("Error:", response.status_code, response.text)