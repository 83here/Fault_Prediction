import requests
import time

# Replace with Laptop A's IP address
server_url = "http://10.0.8.178:5001/get_mse"

class stream:

    def live():    

        while True:
            try:
                response = requests.get(server_url)
                if response.status_code == 200:
                    data = response.json()
                    return (data['mse'],data['prediction'],data['target'])
                else:
                    print("Failed to retrieve MSE. Status code:", response.status_code)
            except requests.exceptions.RequestException as e:
                print("Request failed:", e)

            # Fetch every second
            time.sleep(1)