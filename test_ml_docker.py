import requests

#url = 'https://matrix-pricing-ml-270571767676.us-central1.run.app'
url = 'https://ml-matrix-pricing-270571767676.us-central1.run.app'
url2 = 'https://ml-matrix-price-270571767676.us-central1.run.app'
ticker = 'NVDA'

identity_token = '.----'

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
