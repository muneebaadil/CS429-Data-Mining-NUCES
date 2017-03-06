# -*- coding: utf-8 -*-
import scrapy


class OlxcarscarsSpider(scrapy.Spider):
    name = "OLXcarscars"
    #allowed_domains = ["olx.com.pk/cars/"]
    start_urls = (
        'http://www.olx.com.pk/cars/',
    )

    def parse(self, response):
        urlpieces = response.url.split('=')
        pagenum = '1' if (len(urlpieces) == 1) else urlpieces[-1]

        with open('./OLXcarscars/olxcars%s.html' % pagenum, 'w') as f: 
        	f.write(response.body)
        
        basexpath = '/html/body/div[1]/div[1]/section[2]/div[1]/div/div[2]'
        basexpath = '/html/body/div[1]/div[1]/section[2]/div[1]/div/div[2]'
        nexturlxpath = basexpath + '/span[' + str(int(pagenum)+2) + ']/a/@href'
        nexturl = response.xpath(nexturlxpath).extract_first()

        print 'NEXTURL =', nexturl
        yield scrapy.Request(nexturl)