import requests
import re
import time
from urllib.parse import quote
from bs4 import BeautifulSoup

headers = {
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
}

def read_link(result_list,search_key,page_paragraph_limit):
	wikiUrlPattern = re.compile("^https:\/\/en.wikipedia.org\/wiki\/.*")
	if wikiUrlPattern.match(search_key):
		title = re.sub(r'(^https:\/\/en.wikipedia.org\/wiki\/)','',search_key)
		title = re.sub(r'_',' ',title)
		title = re.sub(r'%27','',title)
		href = search_key
		result_list.append([title, href, read_wiki_page(href, page_paragraph_limit)])

def search_wiki(result_list,search_key,search_list_limit, page_paragraph_limit):
	start_time = time.time()
	url = quote(f'https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search={str(search_key)}',safe='/:?=.&')

	r = requests.get(url, headers=headers)
	print(url)
	html = r.text

	soup = BeautifulSoup(html, 'html.parser')
	[tag.decompose() for tag in soup("style")]
	#print(soup.prettify())

	elements = soup.find_all(class_='mw-search-result')
	#print(str(elements))
	print(f'Finish search on {time.time() - start_time}s')
	for index, item in enumerate(elements):
		if(index>=search_list_limit):
			break
		ahref = item.find(class_='mw-search-result-heading').find('a')
		#print(str(ahref))
		if ahref is not None:
			text = str(ahref)
			#print(text)
			title = re.sub(r'(<[^>]+>)','',text)
			title = re.sub(r'^\s*','',title)
			title = re.sub(r'\t','',title)
			#print(title)
			href = 'https://en.wikipedia.org'+ahref['href']
			#print(href)
			result_list.append([title, href, read_wiki_page(href, page_paragraph_limit)])

def read_wiki_page(url,page_paragraph_limit):
	start_time = time.time()
	r = requests.get(url, headers=headers)
	print(url)
	html = r.text

	soup = BeautifulSoup(html, 'html.parser')
	[tag.decompose() for tag in soup("style")]
	#print(soup.prettify())
	
	elements = soup.find_all(class_='mw-parser-output')
	#print(str(elements))
	print(f'Finish read_wiki_page on {time.time() - start_time}s')
	result_text = ''
	for item in elements:
		p = item.find_all('p')
		#print(str(p))
		for index2, item2 in enumerate(p):
			if(index2>=page_paragraph_limit+1):
				break
			text = str(item2)
			#print(text)
			text = re.sub(r'(<[^>]+>)','',text)
			#print(text)
			text = re.sub(r'(\[[0-9]*\]:*\s*[0-9]*|[\ \n]{2,})','',text)
			#print(text)
			result_text += text
	#print(str(result_text))
	return result_text
