####################
#  This script is used to fetch the designated information from PolyInfo and save to a .csv file ##
####################

#### Tips for Debugging
# 1. Don't remove the website saves points. If the code isn't working, check the .html files to see whether one of the steps is broken. Try deleting the saved .html files and re-running (sometimes this helps)
# 2. Unsure if the API is stable, check via Chrome Dev Tools if API is no longer the same (search Fetch/XHR)
# 3. Make sure to remove the highlighted search parameter for new pages if you are looking at everything other than radiation resistance (or double check the payload for the API call in Chrome Dev Tools)
# 4. Sometimes the API call will fail just for fun... Give it a couple tries and that seems to do the trick. Might be running into a captcha or something on the backend.

import re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import json
import base64

# This function takes in a property (formatted correctly) and a number of polymers, and outputs a .json file for quick access
# I don't understand why but if I remove the website saves the entire thing breaks so DONT REMOVE THEM!!
def fetch_poly_info(property_name, limit): 
    # Login info necessary to access PolyInfo
    email = "kirajm2@illinois.edu"
    password = "Polymer111!"
    AUTH_URL = "https://polymer.nims.go.jp/"
    api = 5974882 # probably not stable?

    ## Attempt to login to PolyInfo and create a session
    ## Site 1
    s = requests.Session()
    site1 = s.get(AUTH_URL, allow_redirects=True)
    soup1 = BeautifulSoup(site1.text, 'html.parser')
    ## Find and access site 2
    link = soup1.find('a', class_='login')['href']
    site2 = s.get(link, allow_redirects=True)
    with open("step2.html", "wb") as f:
        f.write(site2.content)

    ## Find and access site 3
    # Extract SETTINGS from the response HTML. May be unstable...
    settings = re.search(r"var SETTINGS = (\{.*?\});", site2.text, re.DOTALL).group(1)
    csrf_token = re.search(r'"csrf":"([^"]+)"', settings).group(1)
    trans_id = re.search(r'"transId":"([^"]+)"', settings).group(1)
    policy = re.search(r'"policy":"([^"]+)"', settings).group(1)
    tenant_path = re.search(r'"tenant":"([^"]+)"', settings).group(1)
    api_name = re.search(r'"api":"([^"]+)"', settings).group(1)

    link3 = urljoin(
        "https://dicelogin.b2clogin.com",
        f"{tenant_path}/api/{api_name}/selected",
    )
    params = {
        "accountId": "EXGENIdPExchange",
        "csrf_token": csrf_token,
        "tx": trans_id,
        "p": policy,
    }
    site3 = s.get(link3, params=params, allow_redirects=True)
    with open("step3.html", "wb") as f:
        f.write(site3.content)

    ## Find and access site 4
    soup3 = BeautifulSoup(site3.text, "html.parser")
    form3 = soup3.find("form", id="login") # Find the form we need to fill in
    action3 = urljoin(site3.url, form3.get("action"))
    forminp = {}
    for input in form3.find_all("input"):
        name = input.get("name")
        value = input.get("value")
        forminp[name] = value
    forminp["identifier"] = email # fill in the email and password fields
    forminp["password"] = password
    site4 = s.post(action3, data=forminp, allow_redirects=True)
    with open("step4.html", "wb") as f:
        f.write(site4.content)

    ## Now we are logged in and can access data!
    # Submit search to get polymer list
    list_url = urljoin(site4.url, "/PoLyInfo/polymer-list")
    site5 = s.post(list_url, data={"strgkey_uuid": "STRGKEY_POLYMER_LIST_DATA"})
    with open("step5.html", "wb") as f:
        f.write(site5.content)
    # Query the API with search parameters, 
    api_url = urljoin(site4.url, f"/PoLyInfo/api/{api}")
    search_params = {
        "reference": {},
        "limit": str(limit),  # Convert to string
        "offset": 0,
        "property": [{"property_name": property_name}]
    }
    results = s.post(api_url, json=search_params)
    response_data = results.json()
    # Decode from base64
    decoded_results = base64.b64decode(response_data["json"])
    loaded_data = json.loads(decoded_results)
    data = loaded_data
    # Loop through to get the rest of the pages of the query
    offset = limit
    while offset < loaded_data["search_results"]:
        search_params["offset"] = offset

        ###### Weird quirk of the radiation resistance query that does not apply for any other parameter to my knowledge
        if property_name == "Radiation resistance":
            search_params["property"].append({"property_name": property_name})  
        ######

        results = s.post(api_url, json=search_params)
        response_data = results.json()
        decoded_results = base64.b64decode(response_data["json"])
        loaded_data = json.loads(decoded_results)
        data["polymer_data"].extend(loaded_data["polymer_data"])
        offset += limit

    # Save data to .json file    
    with open(f"poly_info_{property_name}.json", "w") as f:
        json.dump(data, f, indent=2)

####################
#  Calling functions
####################
# outputs .json that we want to then parse to csv
fetch_poly_info("Radiation resistance", 50)
# just to demo that we can do more than two pages of query lol
fetch_poly_info("Heat of fusion mol conversion",50) 
