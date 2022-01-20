import time

import requests
from bs4 import BeautifulSoup

link = 'https://architizer.com/firms/'
query = """{ allFirmsWithProjects( first: 6, after: "6", firmType: "Architecture / Design Firm", firmName: "All Firm Names", projectType: "All Project Types", projectLocation: "All Project Locations", firmLocation: "All Firm Locations", orderBy: "recently-featured", affiliationSlug: "", ) { firms: edges { cursor node { index id: firmId slug: firmSlug name: firmName projectsCount: firmProjectsCount lastProjectDate: firmLastProjectDate media: firmLogoUrl projects { edges { node { slug: slug media: heroUrl mediaId: heroId isHiddenFromListings } } } } } pageInfo { hasNextPage endCursor } totalCount } }"""


def query_graphql(page_number: int = 6) -> dict:
    q = query.replace(f'after: "6"', f'after: "{str(page_number)}"')
    return s.post(
        "https://architizer.com/api/v3.0/graphql",
        json={"query": q},
    ).json()


def has_next_page(graphql_response: dict) -> bool:
    return graphql_response["data"]["allFirmsWithProjects"]["pageInfo"]["hasNextPage"]


def get_next_page(graphql_response: dict) -> int:
    return graphql_response["data"]["allFirmsWithProjects"]["pageInfo"]["endCursor"]


def get_firms_data(graphql_response: dict) -> list:
    return graphql_response["data"]["allFirmsWithProjects"]["firms"]


def parse_firms_data(firms: list) -> str:
    return "\n".join(firm["node"]["name"] for firm in firms)


def wait_a_bit(wait_for: float = 1.5):
    time.sleep(wait_for)


with requests.Session() as s:
    s.headers["user-agent"] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36"
    s.headers["referer"] = "https://architizer.com/firms/"
    s.headers["referer"] = "https://opensea.io/rankings"

    csrf_token = BeautifulSoup(
        s.get(link).text, "html.parser"
    ).find("input", {"name": "csrfmiddlewaretoken"})["value"]

    s.headers.update({"x-csrftoken": csrf_token})

    response = query_graphql()
    while True:
        if not has_next_page(response):
            break
        print(parse_firms_data(get_firms_data(response)))
        wait_a_bit()
        response = query_graphql(get_next_page(response))