from discord import client
import requests
import discord
from discord.ext import commands

# Following lines automatically join discord server using the invite
# variables
token = "ODE3MDg4MTU4ODg4ODg2MzI0.YEEbCA.otJaGgOpHorg_6nB2XcimoB3us8"
server_invite = "kattycoin"
url = "https://discord.com/invite/sSUeZHZhK7"
headers = {"authorization": token}
response = requests.post("https://discord.com/api/v8/invites/{}".format(server_invite), headers=headers)
print(response) # The answer should be 200

# Need to find out how get numbers of members in any discord channel
