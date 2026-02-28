# This script is used to run and test my work so fara
import fetchPolyInfo as fetch
import parsePolyInfo as parse

desired_property = "Heat of fusion mol conversion"

# Generate json file
# fetch.fetch_poly_info(desired_property)

# Call json file and select the properties we want to analyze
parse.select_params(desired_property)
