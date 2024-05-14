import requests

url = "https://maestro.dadosfera.ai/auth/sign-in"

payload = {
    "username": "mikael.akihitto98@gmail.com",
    "password": "Ramise@2011"
}
headers = {
    "accept": "application/json",
    "dadosfera-lang": "pt-br",
    "content-type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)