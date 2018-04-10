import requests
from bs4 import BeautifulSoup

def get_bs(url):
    req = requests.get(url)
    assert req.status_code == 200, f'Request status {req.status_code} on {url}.'
    return BeautifulSoup(req.content, 'html5lib')

