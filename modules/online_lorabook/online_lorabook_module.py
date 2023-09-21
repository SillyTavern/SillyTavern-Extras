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
import re
import time
import json

if __name__ == '__main__':
	from parser import wikipedia as wikipedia
	from parser import fandom as fandom
else:
	from .parser import wikipedia as wikipedia
	from .parser import fandom as fandom

headers = {
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
}

def search(search_key,search_list_limit,page_paragraph_limit,search_site):
	result_list = []
	
	#read page url without search
	wikipedia.read_link(result_list,search_key,page_paragraph_limit)
	if len(result_list) > 0:
		return result_list
	
	fandom.read_link(result_list,search_key,page_paragraph_limit)
	if len(result_list) > 0:
		return result_list

	#or else searh the page first
	if search_site == "wikipedia":
		wikipedia.search_wiki(result_list,search_key,search_list_limit, page_paragraph_limit)
	else:
		fandom.search_fandom(result_list,search_key,search_list_limit, page_paragraph_limit)
				
	#print(str(result_list))
	return result_list

def run(text,params):
	search_list_limit=1
	if(params.get("search_list_limit") != None):
		search_list_limit = params["search_list_limit"]

	page_paragraph_limit=1
	if(params.get("page_paragraph_limit") != None):
		page_paragraph_limit = params["page_paragraph_limit"]

	search_site='wikipedia'
	if(params.get("search_site") != None):
		search_site = params["search_site"]

	start_time = time.time()

	search_keys = re.findall(r'\"\"[^\"\"]*\"\"', str(text))
	print(str(search_keys))

	entries = []
	for search_key in search_keys:
		search_result_list = search(re.sub(r'\"\"','',search_key),search_list_limit,page_paragraph_limit,search_site)
		for item in search_result_list:
			search_result_title = item[0]
			search_result_href = item[1]
			search_result_text = item[2]
			entries.append({
				'text':f'{search_result_text}', 
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
	#text = 'Did you hear about ""LK99""? It is a new ""materials"", checkout here ""https://en.wikipedia.org/wiki/Ambient_pressure""".'
	text = '""Shadowheart"" is my favorite character in ""Baldur\'s Gate 3"", checkout here ""https://forgottenrealms.fandom.com/wiki/Lae%27zel"".'
	params = {
		"search_list_limit" : 1,
		"page_paragraph_limit": 1,
		"search_site": 'wikipedia',
		#"search_site": 'fandom',
	}

	run(text,params)
