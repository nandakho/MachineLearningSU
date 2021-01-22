from bahasa.stemmer import Stemmer
import requests
import re
from bs4 import BeautifulSoup

# Ambil html page dari url
url = "https://finance.detik.com/berita-ekonomi-bisnis/d-5345097/ditunggu-tunggu-masyarakat-blt-pegawai-kapan-cair"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# Find article from html dan buang html tag
articleDirty = str(soup.find_all(class_='detail__body-text itp_bodycontent'))
clean = re.compile('<.+?>')
articleClean = re.sub(clean, '', articleDirty)

# Convert artikel ke kata dasar
stemmer = Stemmer()
print(stemmer.stem(articleClean))