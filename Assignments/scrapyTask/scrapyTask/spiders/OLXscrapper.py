# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup

def findcategories(tag):
    return unicode(tag.string).find('View all Ads') != -1 and tag.name == 'span'

class OlxcarscarsSpider(scrapy.Spider):
    name = "OLXcarscars"
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
    	pass