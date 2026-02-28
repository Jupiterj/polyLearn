####################
#  This script is used to fetch the designated information from PolyInfo and save to a .csv file ##
####################

#### Tips for Debugging
# 1. Don't remove the website saves points. If the code isn't working, check the .html files to see whether one of the steps is broken. Try deleting the saved .html files and re-running (sometimes this helps)
# 2. Unsure if the API is stable, check via Chrome Dev Tools if API is no longer the same (search Fetch/XHR)
# 3. Make sure to remove the highlighted search parameter for new pages if you are looking at everything other than radiation resistance (or double check the payload for the API call in Chrome Dev Tools)
# 4. Sometimes the API call will fail just for fun... Give it a couple tries and that seems to do the trick. Might be running into a captcha or something on the backend. I added a reauth for the individually 

import re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import json
import base64
import time

# update this based on new logins
def getlogin():
    email = "jadenjpt@stanford.edu"
    password = "wEpmiv-nevcip-4pahvy"

    return email, password

# This function performs the authentication process necessary to open the PolyInfo database
# I don't understand why but if I remove the website saves the entire thing breaks so DONT REMOVE THEM!!
def authenticate(email, password, AUTH_URL="https://polymer.nims.go.jp/"):
    # Re-authenticate if we get caught in a captcha
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
    print("Accessing site 2")
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
    print("Accessing site 3")
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
    print("Accessing site 4")
    return s, site4


# This function takes in a property (formatted correctly) and a number of polymers, and outputs a .json file for quick access
def fetch_poly__list_info(property_names, limit=50): 
    
    start_time = time.time()
    AUTH_URL = "https://polymer.nims.go.jp/"
    list_api = 5974882 # probably not stable?
    email, password = getlogin()
    s, site4 = authenticate(email, password)
    max_retries = 3
    retry_count = 0
    success = 0
    # parse property names
    count = 0
    prop_list = []
    while count < len(property_names):
        prop_list.append({"property name": property_names[count]})
        count = count + 1

    while retry_count < max_retries and not success:
        try: 
            # Submit search to get polymer list
            list_url = urljoin(site4.url, "/PoLyInfo/polymer-list")
            site5 = s.post(list_url, data={"strgkey_uuid": "STRGKEY_POLYMER_LIST_DATA"})
            with open("step5.html", "wb") as f:
                f.write(site5.content)
            # Query the API with search parameters, 
            api_url = urljoin(site4.url, f"/PoLyInfo/api/{list_api}")
            search_params = {
                "reference": {},
                "limit": str(limit),  # Convert to string
                "offset": 0,
                "property": prop_list
            }
            print("Performing list query")
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
                if "Radiation resistance" in property_names:
                    search_params["property"].append({"property_name": "Radiation resistance"})  
                ######

                results = s.post(api_url, json=search_params)
                response_data = results.json()
                decoded_results = base64.b64decode(response_data["json"])
                loaded_data = json.loads(decoded_results)
                data["polymer_data"].extend(loaded_data["polymer_data"])
                offset += limit
            success = True
            print("List query successful")
        except Exception as e:
            time.sleep(5)
            retry_count += 1
            print(f"Query failed. Re-authenticating (try {retry_count})...")
            s, site4 = authenticate(email, password, AUTH_URL)

    # Save data to .json file    
    with open(f"poly_info_{property_names[0]}.json", "w") as f:
        json.dump(data, f, indent=2)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"This successful query took {elapsed_time:.2f} seconds")

# The function fetches SMILES data individually from PolyInfo, when inputted the polymer list data
def fetch_poly_smiles_info(file_name):
    # Unfortunately, we need to physically click into every polymer to get its SMILES representation. This means it is very easy to encounter a captcha. Just rerun it a couple times and it generally works....
    email, password = getlogin()
    item_api = 6627766
    s, site4 = authenticate(email, password)
    with open(file_name, "r") as file:
        data = json.load(file)
    # Query the API with search parameters, 
    api_url = urljoin(site4.url, f"/PoLyInfo/api/{item_api}")
    print("Performing individual polymer query")
    for i in range(len(data["polymer_data"])):
        max_retries = 3
        retry_count = 0
        success = False
        while retry_count < max_retries and not success:
            try: 
                # Weird quirk where they name the UUIDs of homo and copolymers different things
                if "polymer_uuid" in data["polymer_data"][i]:
                    search_params = {
                        "pid_uuid": data["polymer_data"][i]["polymer_uuid"]
                    }
                elif "copolymer_uuid" in data["polymer_data"][i]:
                    success = True
                    continue
                else:
                    success = True
                    continue
                results = s.post(api_url, json=search_params)
                response_data = results.json()
                # Decode from base64
                decoded_results = base64.b64decode(response_data["json"])
                loaded_data = json.loads(decoded_results)

                # Some polymers will not have SMILES or formula weights
                if "formula_weight" in loaded_data:
                    data["polymer_data"][i]["formula_weight"] = loaded_data["formula_weight"]
                if "smiles" in loaded_data:
                    data["polymer_data"][i]["smiles"] = loaded_data["smiles"]
                print(f"Successful query of polymer {i+1}, moving on")
                success = True
            except Exception as e:
                # Attempt to reauthenticate if error is thrown
                retry_count += 1
                print(f"Error occured on polymer {i+1}, query attempt {retry_count}: {e}")
                if retry_count < max_retries:
                    time.sleep(2)
                    s, site4 = authenticate(email, password)
                else:
                    print(f"Skipping polymer {i}, no access granted")
    
    # Save data to .json file      
    with open(f"new_smiles.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Individual polymer query successful")

fetch_poly_smiles_info("poly_info_Glass transition temperature.json")