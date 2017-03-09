# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup

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
        tds = soup.find_all(name='td')
        for tag in tds: 
        	nexturl = tag.find(name='a', href=True)['href']
        	yield scrapy.Request(nexturl, callback=self.parseCategory)

    def parseCategory(self, response):
    	pass
