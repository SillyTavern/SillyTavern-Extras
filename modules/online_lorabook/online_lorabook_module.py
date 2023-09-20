"""
online_lorabook module for SillyTavern Extras

Authors:
    - donlinglok (https://github.com/donlinglok)
	- <Add your name here> ()

References:
    - Code adapted from:
        - LemonQu-GIT/ChatGLM-6B-Engineering https://github.com/LemonQu-GIT/ChatGLM-6B-Engineering
        - THUDM/WebGLM https://github.com/THUDM/WebGLM
"""
import sys
import requests
import re
import time
import json
from urllib.parse import quote
from bs4 import BeautifulSoup

headers = {
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
}

def read_wiki_page(url,wiki_page_paragraph_limit):
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
			if(index2>=wiki_page_paragraph_limit+1):
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

def search_wiki_list(search_key,wiki_search_list_limit,wiki_page_paragraph_limit):
	start_time = time.time()
	result_list = []
	
	wikiUrlPattern = re.compile("^https:\/\/en.wikipedia.org\/wiki\/.*")
	if wikiUrlPattern.match(search_key):
		title = re.sub(r'(^https:\/\/en.wikipedia.org\/wiki\/)','',search_key)
		title = re.sub(r'_',' ',title)
		href = search_key
		result_list.append([title, href, read_wiki_page(href, wiki_page_paragraph_limit)])
		return result_list

	url = quote(f'https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search={str(search_key)}',safe='/:?=.&')

	r = requests.get(url, headers=headers)
	print(url)
	html = r.text

	soup = BeautifulSoup(html, 'html.parser')
	[tag.decompose() for tag in soup("style")]
	#print(soup.prettify())

	elements = soup.find_all(class_='mw-search-result')
	#print(str(elements))
	print(f'Finish search_wiki_list on {time.time() - start_time}s')
	for index, item in enumerate(elements):
		if(index>=wiki_search_list_limit):
			break
		# old version
		#extiw = item.find(class_='extiw')
		#print(str(extiw))
		#if extiw is not None:
		#	text = str(extiw)
		#	print(text)
		#	title = re.sub(r'(<[^>]+>)','',text)
		#	href = extiw['href']
		#	result_list.append([title, href])

		extiw2 = item.find(class_='mw-search-result-heading').find('a')
		#print(str(extiw2))
		if extiw2 is not None:
			text = str(extiw2)
			#print(text)
			title = re.sub(r'(<[^>]+>)','',text)
			href = 'https://en.wikipedia.org'+extiw2['href']
			result_list.append([title, href, read_wiki_page(href, wiki_page_paragraph_limit)])
			
	#print(str(result_list))
	return result_list

def run(text,wiki_search_list_limit,wiki_page_paragraph_limit):
	start_time = time.time()

	search_keys = re.findall(r'\"\"[^\"\"]*\"\"', str(text))
	print(str(search_keys))

	entries = []
	for search_key in search_keys:
		search_wiki_result_list = search_wiki_list(re.sub(r'\"\"','',search_key),wiki_search_list_limit,wiki_page_paragraph_limit)
	
		for item in search_wiki_result_list:
			search_result_title = item[0]
			search_result_href = item[1]
			search_result_text = item[2]
			entries.append({
				'text':f'{search_result_title} is {search_result_text}', 
				'contextConfig':
				{
					'prefix':'', 
					'suffix':f'\n^{search_result_title}({search_result_href})',
					'tokenBudget':2048,
					'reservedTokens':0,
					'budgetPriority':400,
					'trimDirection':'trimBottom',
					'insertionType':'newline',
					'insertionPosition':-1
				},
				'displayName':f'{search_result_title}',
				'lastUpdatedAt':int(time.time()),
				'keys':[f'{search_result_title}'],
				'searchRange':1000,
				'enabled':True,
				'forceActivation':False
			})
			
	result = json.dumps({
		'lorebookVersion':1,
		'entries':entries
		}, ensure_ascii=False)
	print(result)

	print(f'Finish run on {time.time() - start_time}s')
	return result

if __name__ == '__main__':
	text = 'Did you hear about ""LK99""? It is a new ""materials"", checkout here ""https://en.wikipedia.org/wiki/Ambient_pressure""".'
	if len(sys.argv) > 1:
		text = str(sys.argv[1])

	wiki_search_list_limit = 1
	if len(sys.argv) > 2:
		wiki_search_list_limit = int(sys.argv[2])

	wiki_page_paragraph_limit = 1
	if len(sys.argv) > 3:
		wiki_page_paragraph_limit = int(sys.argv[3])

	run(text,wiki_search_list_limit,wiki_page_paragraph_limit)