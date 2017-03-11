# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup
import re 

def findcategories(tag):
    return unicode(tag.string).find('View all Ads') != -1 and tag.name == 'span'

class OlxscraperSpider(scrapy.Spider):
    name = "OLXscraper"
    #allowed_domains = ["olx.com.pk/cars/"]
    start_urls = (
        'http://www.olx.com.pk/',
    )

    def parse(self, response):
    	"""
    	Home page (olx.com.pk) parser to get to a page containing multiple 
    	categories.
    	"""
        html = response.body
        soup = BeautifulSoup(html, 'lxml')

        tag = soup.find(name='td')
        nexturl = tag.find(name='a', href=True)['href']
        yield scrapy.Request(nexturl, callback=self.parseCategory)
 
    def parseCategory(self, response):
    	"""
		Category page parser. It finds out each category ads and forwards
		each category-related ads list to parseList() 
    	"""
    	html = response.body
    	soup = BeautifulSoup(html, 'lxml')
    	
    	cats = soup.find_all(findcategories)
    	for c in cats: 
    		nexturl = c.parent['href']
    		nexturl = nexturl + '?search[photos]=false'
    		yield scrapy.Request(nexturl, callback=self.parseList)

    def parseList(self, response):
    	"""
		Parses a list page; forwards each ad on the list to parseAd(); and 
		recursively calls itself on the next list page. 
    	"""
    	#Extracting html off the response and making its soup.. 
    	html = response.body
    	soup = BeautifulSoup(html, 'lxml')

    	#Getting a list of ads' tags. 
    	container = soup.find('div', id='listContainer')
    	section = container.find('section', id='body-container')	
    	ads = section.find_all('td', class_='offer onclick ')

    	for ad in ads:
    		#Generating a new request to parse an individual ad. 
    		nexturl = ad.find('a', href=True)['href']
    		category = container.section.div.div.div.find_all('li')[-1].h1.string
    		
    		req = scrapy.Request(nexturl, callback=self.parseAd) 
    		#Indirectly passing an argument to parseAd() 
    		req.meta['category'] = category
    		yield req 

    	#Extacting the next page's link and recursively calling itself
    	#on next page's list (if there's any). 
    	pagertag = soup.find(name='div', class_='pager')
    	nextpagetag = pagertag.find(name='span', class_=['next'])
    	if nextpagetag is not None: 
    		nexturl = nextpagetag.a['href']
    		yield scrapy.Request(nexturl, callback=self.parseList)

    def parseAd(self, response):
    	"""
		Finally, parse an individual ad and stores its information. 
    	"""
    	#Extracting html off response and making its soup
    	html = response.body 
    	soup = BeautifulSoup(html, 'lxml')
    	
    	#Dividing the page into sections for future convenience.
    	left = soup.find(name='div', class_=['offercontent'])
    	right = soup.find(name='div', class_=['offerbox'], id=['offerbox'])
    	
    	#Left section attributes. 
    	title = unicode(left.find(name='h1', class_=True).string)
    	loc = unicode(left.div.div.p.span.strong.string)
    	adID = left.div.div.p.small.span.find('span', class_=['rel']).string
    	views = int(left.find(name='div', id='offerbottombar').strong.string)

    	#Right section attributes.
    	pricebox = right.div.div.div.find('div', class_=['pricelabel'])
    	price = int(re.sub(r'[^\d]', '', pricebox.strong.string)) if (pricebox is not None) else None

    	userdatabox = right.div.div.div.find('div', class_='userdatabox')
    	user = unicode(userdatabox.p.span.string) if (userdatabox is not None) else None

    	contactbox = right.div.div.div.find('div', class_=['contactbox'])
    	contact = contactbox.strong.string if (contactbox is not None) else None
    	
    	yield {
    	'title': re.sub(pattern=r'\s', repl='', string=title),
    	'URL': response.url, 
    	'loc': re.sub(pattern=r'\s', repl='', string=loc),
    	'ID': re.sub(pattern=r'\s', repl='', string=adID),
    	'views': views,
    	'price': price, 
    	"user name": user, 
    	'contact': contact,
    	'category': response.meta['category']
    	}
    	return 