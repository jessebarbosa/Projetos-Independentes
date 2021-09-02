# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:36:13 2021

@author: roje0001
"""

import re, time
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
   
#extrai dados
def get_data(url):
    #instância do driver
    driver = webdriver.Chrome('drivers/chromedriver')
    driver.get(url)
    
    time.sleep(3)
    
    #clica no elemento "select"
    xpath_select = '/html/body/div[1]/div[3]/div/div[1]/div/div[1]/div[3]/div[1]/div[2]/span/select'
    select = driver.find_element_by_xpath(xpath_select)
    webdriver.ActionChains(driver).click(select).perform()
    
    time.sleep(3)
    
    #clica no elemento "option" para selecionar as avaliações mais recentes
    mais_recente  = driver.find_element_by_partial_link_text('Mais recentes')
    webdriver.ActionChains(driver).click(mais_recente).perform()
    
    time.sleep(3)
    
    master = []
    s = 0
    for i in range(2,7):
        #clica no elemento "select"
        xpath_select_rating = '/html/body/div[1]/div[3]/div/div[1]/div/div[1]/div[3]/div[2]/div[2]/div[2]/span/select'
        select_rating = driver.find_element_by_xpath(xpath_select_rating)
        webdriver.ActionChains(driver).click(select_rating).perform()
        
        time.sleep(3)
        
        stars = 7-i
        
        #seleciona a quantidade de estrelas das avaliações
        xpath_rating = '/html/body/div[3]/div/div/ul/li[{}]'.format(i)
        rating = driver.find_element_by_xpath(xpath_rating)
        webdriver.ActionChains(driver).click(rating).perform()
        
        time.sleep(10)
        
        page_master = []
        
        #coleta as avaliações
        xpath_review = '/html/body/div[1]/div[3]/div/div[1]/div/div[1]/div[5]/div[3]/div/div/div/div/div[4]'
                        #/html/body/div[1]/div[3]/div/div[1]/div/div[1]/div[5]/div[3]/div/div[11]/div/div/span
        reviews = driver.find_elements_by_xpath(xpath_review)
        reviews  = [[r.text, stars] for r in reviews]
        page_master.append(reviews)
        
        s += len(reviews)
        print(s)
        
        #vai para próxima página
        lst_proximo = driver.find_elements_by_partial_link_text('Próximo')
        
        while len(lst_proximo) > 0:
            proximo = lst_proximo[0]
            webdriver.ActionChains(driver).click(proximo).perform()
            time.sleep(5)
            
            xpath_review = '/html/body/div[1]/div[3]/div/div[1]/div/div[1]/div[5]/div[3]/div/div/div/div/div[4]'
            reviews = driver.find_elements_by_xpath(xpath_review)
            reviews  = [[r.text, stars] for r in reviews]
            page_master.append(reviews)
            
            s += len(reviews)
            print(s)
            
            lst_proximo = driver.find_elements_by_partial_link_text('Próximo')

        for page in page_master:
            for review in page:
                master.append(review)
            
        time.sleep(3)
        
    df = pd.DataFrame(master, columns = ['Review', 'Rating'])
    
    return df

