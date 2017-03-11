# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup
import re 

def findcategories(tag):
    return unicode(tag.string).find('View all Ads') != -1 and tag.name == 'span'

def mainbox(tag):
	return tag.name == 'div' and tag.has_attr('class') and \
	('clr' in tag['class']) and ('offerbody' in tag['class'])

class OlxscraperSpider(scrapy.Spider):
    name = "OLXscraper"
    #allowed_domains = ["olx.com.pk/cars/"]
    start_urls = (
        'http://www.olx.com.pk/',
    )

    #Home page (olx.com.pk) parser to get different categories pages.
    def parse(self, response):
        html = response.body
        soup = BeautifulSoup(html, 'lxml')
        tag = soup.find(name='td')
        nexturl = tag.find(name='a', href=True)['href']
        yield scrapy.Request(nexturl, callback=self.parseCategory)

    
    def parseCategory(self, response):
    	html = response.body
    	soup = BeautifulSoup(html, 'lxml')
    	cats = soup.find_all(findcategories)
    	for c in cats: 
    		nexturl = c.parent['href']
    		nexturl = nexturl + '?search[photos]=false'
    		yield scrapy.Request(nexturl, callback=self.parseList)

    def parseList(self, response):
    	html = response.body
    	soup = BeautifulSoup(html, 'lxml')

    	container = soup.find('div', id='listContainer')
    	section = container.find('section', id='body-container')	
    	ads = section.find_all('td', class_='offer onclick ')

    	for ad in ads:
    		nexturl = ad.find('a', href=True)['href']
    		yield scrapy.Request(nexturl, callback=self.parseAd) 

    def parseAd(self, response):
    	html = response.body 
    	soup = BeautifulSoup(html, 'lxml')
    	mainsec = soup.find(mainbox)
    	left, right = mainsec.find_all(name='div', recursive=False)
    	
    	title = unicode(left.find(name='h1', class_=True).string)
    	loc = unicode(left.div.div.p.span.strong.string)
    	adID = left.div.div.p.small.span.find('span', class_=['rel']).string

    	yield {
    	'title': re.sub(pattern=r'\s', repl='', string=title),
    	'URL': response.url, 
    	'loc': re.sub(pattern=r'\s', repl='', string=loc),
    	'ID': re.sub(pattern=r'\s', repl='', string=adID)
    	}
    	return 