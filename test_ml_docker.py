import requests

#url = 'https://matrix-pricing-ml-270571767676.us-central1.run.app'
url = 'https://ml-matrix-pricing-270571767676.us-central1.run.app'
url2 = 'https://ml-matrix-price-270571767676.us-central1.run.app'
ticker = 'NVDA'

identity_token = 'eyJhbGciOiJSUzI1NiIsImtpZCI6IjVkMTJhYjc4MmNiNjA5NjI4NWY2OWU0OGFlYTk5MDc5YmI1OWNiODYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTA0NDg5MDE5NjY5NDI0MDU5ODEyIiwiZW1haWwiOiJteWtvbGEuemh1cmF1c2t5eUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXRfaGFzaCI6IjIwQmd5TEFuUTlLcmszdzM5TERBTUEiLCJpYXQiOjE3Mzk4MjkwODgsImV4cCI6MTczOTgzMjY4OH0.X7Rvya5bSp3ai6frEsVvYFv77QmcVKiNx9gXcOHTQxWZkESb8XfadvxLxgqofRjM-12y9w9n5vlAHpGnGLCcPPGG-M06eYO_yebnLaTBNuZwsqLa6mxs7ddnnaUkRWYdShqM1nXOCkqcSnUBASeJnePliyZvWsT3Q6wN7o87fNkQ4qrAebJXsO7DmNX7cnuabTjtMJq7WJ67nMkph3mbcVOMb2smna-pb2fWZdNHf2qu_byNFTQ0mKuWOQUiAEpotvq7bwL2jqKIzA2C6fSD8XhjjOz_3oA4-qktt8Jvq1xLabJqSNW18igpr4eHeYWorVXNvi8OIVeMW2g5PcSIwg'

payload = {
    "Ticker": ticker
}

headers = {
    "Authorization": "Bearer "+str(identity_token),
    "Content-Type": "application/json"
}

#r = requests.post(url2, json=payload, headers=headers)
r2 = requests.post(url, json=payload)

#print (r.text)
print (r2.text)