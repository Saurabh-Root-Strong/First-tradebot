# Import the required module from the fyers_apiv3 package
from fyers_apiv3 import fyersModel

# Define your Fyers API credentials
client_id = "WVDZUTO6HL-100"
secret_key = "QEUPA8AAA6"
redirect_uri = "http://127.0.0.1:8085"  # Replace with your redirect URI
response_type = "code" 
grant_type = "authorization_code"  

# The authorization code received from Fyers after the user grants access
auth_code = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJXVkRaVVRPNkhMIiwidXVpZCI6ImJiZjE4Y2YyYTQ4MDQ3ZDRiMDU2NDJjYjNkNzM2NGFlIiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IlhTMDQxOTUiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiJiMzdlZTJkM2E1MWU3NWUwNjM2NjU4MDM3ZTIzMjRmZWViNjU4Zjg0ZWUxODNjMjdhNTU3MTdkZiIsImlzRGRwaUVuYWJsZWQiOiJZIiwiaXNNdGZFbmFibGVkIjoiTiIsImF1ZCI6IltcImQ6MVwiLFwiZDoyXCIsXCJ4OjBcIixcIng6MVwiLFwieDoyXCJdIiwiZXhwIjoxNzgwNDk5ODQ5LCJpYXQiOjE3ODA0Njk4NDksImlzcyI6ImFwaS5sb2dpbi5meWVycy5pbiIsIm5iZiI6MTc4MDQ2OTg0OSwic3ViIjoiYXV0aF9jb2RlIn0.vIjduEOh5A0KsnfT005E4ztk09n2URwORBUxfbwWGFw"

# Create a session object to handle the Fyers API authentication and token generation
session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key, 
    redirect_uri=redirect_uri, 
    response_type=response_type, 
    grant_type=grant_type
)

# Set the authorization code in the session object
session.set_token(auth_code)

# Generate the access token using the authorization code
response = session.generate_token()

# Print the response, which should contain the access token and other details
print(response)

