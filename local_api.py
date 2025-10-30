import requests

# Send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000") 

# Print the status code
print("Status Code:", r.status_code)
# Print the welcome message
print("Response Body:", r.json())


# Define the data to send in the POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST using the data above
url = "http://127.0.0.1:8000/data/"
r = requests.post(url, json=data)   # Your code here

# Print the status code
print("Status Code:", r.status_code)
# Print the result
print("Response Body:", r.json())
