from discord import client
import requests
import discord
from discord.ext import commands
from discord import *

# Following lines automatically join discord server using the invite
# variables
token_invite = "ODE3MDg4MTU4ODg4ODg2MzI0.YEEbCA.otJaGgOpHorg_6nB2XcimoB3us8"
server_invite = "kattycoin"
url = "https://discord.com/invite/sSUeZHZhK7"
headers = {"authorization": token_invite}
response = requests.post("https://discord.com/api/v9/invites/{}".format(server_invite), headers=headers)
print(response) # The answer should be 200

# Need to find out how get numbers of members in any discord channel
token_bot = "OTMwNTIwMTIxOTM0NDg3Njkz.Yd3EZQ.WDlOckCD704xJcHLX7oue6PXFOU"
