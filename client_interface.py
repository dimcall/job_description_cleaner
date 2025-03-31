import requests
# URL of the server
url = "http://localhost:8000/predict"
# Data to be sent to the server
data = [
    {
        "html": "<p><strong>Ihre Hauptaufgaben:</strong></p><ul><li>Diverse Arbeiten inder Produktion (stehende Tätigkeit)</li><li>4-Schichtbetrieb: 06.00-14.00/14.00-22.00/22.00-06.00 Uhr <strong>inkl. Wochenende</strong></li></ul><p> </p><p><strong>Ihr fachliches Profil:</strong></p><ul><li>Bereitschaft für <strong>jede Schicht zwingend</strong></li><li>Wohnort im Raum Baden/Zurzach zwingend<strong>!</strong></li><li>Führerausweis Kategorie B undeigenes Auto</li><li>Produktionserfahrung erwünscht</li><li>Alter zwischen 18 und 50 Jahren</li><li><strong>GuteDeutsch Kenntnisse</strong> ZWINGEND!!!</li><li></li></ul><p> </p><p><span>Bewerbungen ausserhalb des Kantons Aargau/Zürich-Limmattal werden nicht bearbeitet.</span></p><p> </p><p><strong> </strong></p>" #4607
    },
    {
        "html": "<p><strong> <span>Main Responsibilities:</span></strong></p><ul><li><span>Provide regulatorystrategy for the MENA region;</span></li><li><span>Works with some independence under limited supervision to provide strategic and operational regulatory direction;</span></li><li><span>Lead and coordinate timely and highquality preparation of allnecessary supporting documentation by internal or external experts for MENA license renewals, MENA variations (except CMC-related variations), PSUR submissions for assigned registered products;</span></li><li><span>Interact with the MENA and other Rest of World Health Authorities to solve regulatory issues related to assigned registered products;</span></li><li><span>Represent Regulatory Affairs (RA) in negotiations with HA in MENA;</span></li><li><span>Provide prompt and complete responses to regulatory relevant queries from various stakeholders relating to assigned brands;</span></li><li><span>Coordinate archiving of submitted registration dossiers and other relevant Health Authority communications;</span></li><li><span>Evaluate and review regulatory parts of contracts and CRO quotations.</span></li></ul><p><strong> <span>Qualifications and Experience:</span></strong></p><ul><li><span>Degree in life science preferable and at least 6 years experience in regulatory affairs;</span></li><li><span>Sound experience with MENA (Middle East and North Africa);</span></li><li><span>Ability to work in multicultural environment;</span></li><li><span>Excellent communication and interpersonal skills;</span></li><li><span>Accuracy and good project management skills;</span></li><li><span>Excellent command of spoken and written English;</span></li><li><span>Relevant working/residency permit or Swiss/EU-Citizenship required.</span></li></ul>" #2274
    },
    {
        "html": "<p>Formation debase technique, CFC de polym&eacute;canicien ou de dessinateur-machines&nbsp;</p> <p>Titulaire d'un Bachelor en m&eacute;canique ou micro-m&eacute;canique ou &eacute;quivalent</p> <p>Exp&eacute;rience professionnelleminimale 2 ans</p> <p>Profil junior ou senior bienvenu</p> <p>Langues : FR/ALL courant, ANG un avantage</p>" #2509
    },
    {
        "html": """Responsabile finanze e membro della Direzione del gruppo
                80-100%, Berna    Creare insieme nuove prospettive: promuovete ulteriormente lo sviluppo di uno dei maggiori datori di lavoro della Svizzera con impegno ed entusiasmo e sfruttate la diversificazione dei profili professionali e il margine di azione creativo per la vostra carriera individuale.   Dinamizzate il giallo insieme a noi
                e inviate la vostra candidatura.         La sua area di attivit
                Come membro della Direzione del gruppo, contribuisce a un ottimale processo decisionale dal punto di vista finanziario e strategico a livello di gruppo. Sensibilizza i membri della Direzione del gruppo per questioni relative alla gestione finanziaria e valuta tutti i progetti con implicazioni finanziarie della Direzione del gruppo e del Consiglio di amministrazione.  in grado di gestire e chiarire gli aspetti di tutte le questioni finanziarie in funzione dei vari livelli, nonch di esporre le loro conseguenze e ripercussioni.
                Come Responsabile finanze del gruppo cura i mandati presso le societ del gruppo e le ditte di terzi quale rappresentante della Posta. Insieme alla direttrice generale, partecipa alle sedute del Consiglio di amministrazione di Posta e alle commissioni CdA finanziariamente rilevanti quale rappresentante fisso. Inoltre,  responsabile della preparazione e dell'impiego di strumenti di gestione finanziaria (pianificazione finanziaria, budget, rapporti, statistiche, allestimento dei conti), nonch del controlling dell'intera azienda. Provvede a una gestione del rischio adeguata e a una copertura assicurativa ottimale ed  responsabile delle imposte dell'azienda. Stabilisce il processo per acquisizioni e fusioni.  responsabile di programmi di sviluppo strategici in ambito finanziario come ad es. lo sviluppo di Shared Services Finanze e Organizzazione acquisti.
                A livello tecnico gestisce inoltre i responsabili Finanze e contabilit delle unit aziendali e di gestione.      Il suo profilo
                - Vanta una pluriennale esperienza in ambito dirigenziale e dispone di competenze nell'ambito finanziario, preferibilmente acquisite in una grande azienda
                - Ha conseguito un diploma universitario o di formazione equivalente e possiede provate nozioni approfondite di economia aziendale
                -  esperto/a di consulenza ai responsabili delle decisioni della Direzione del gruppo e del Consiglio di amministrazione e grazie alle sue spiccate capacit comunicative riesce a riassumere temi finanziari complessi in funzione dei destinatari e dei vari livelli
                -  dotato/a di spirito di iniziativa ed elabora da diversi punti di vista soluzioni sostenibili
                -  considerato/a un/una persona di contatto competente e orientata alle soluzioni
                - Madrelingua tedesca o francese con buona conoscenza delle altre lingue nazionali, nonch ottima padronanza della lingua inglese
                Saremo lieti di ricevere il suo dossier di candidatura completo, che la preghiamo di spedire per posta o per e-mail al consulente Executive Search da noi incaricato, il Sig. Guido Schilling, Guido Schilling AG, Prime Tower, Hardstrasse 201, 8005 Zurigo (guido.schilling@guidoschilling.ch). Il termine per l'invio delle candidature  il 7 gennaio 2016.        Dettaglio d'impiego
                Impiego:80-100%
                Luogo (luoghi) d'impiego:Berna
                Numero di riferimento:50853373
                Tipo di lavoro:Contr. di durata indeterminata
                Unit della Posta:Finanze      Il suo contatto
                Guido Schilling
                Executive Search
                Telefon +41 44 366 63 33
                - Condividi Condividi su | http://www.facebook.com/share.php?u=http://direktlink.prospective.ch/?view=D3C55E38-BCCA-457E-B67A3F7A44500759 | https://plus.google.com/share?url=http://direktlink.prospective.ch/?view=D3C55E38-BCCA-457E-B67A3F7A44500759 | http://twitter.com/?status=%23Post: Responsabile%20finanze%20e%20membro%20della%20Direzione%20del%20gruppo http://direktlink.prospective.ch/?view=D3C55E38-BCCA-457E-B67A3F7A44500759 | http://www.linkedin.com/shareArticle?mini=true&url=http://direktlink.prospective.ch/?view=D3C55E38-BCCA-457E-B67A3F7A44500759 | https://www.xing.com/app/user?op=share;url=http://direktlink.prospective.ch/?view=D3C55E38-BCCA-457E-B67A3F7A44500759
                - Stampa""" #11365
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