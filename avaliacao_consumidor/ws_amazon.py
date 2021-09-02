# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:03:41 2021

@author: roje0001
"""
from ws_functions import get_data
import pandas as pd

# lista de urls de celulares da fabricante xiaomi
lst_url = ['https://www.amazon.com.br/Celular-Xiaomi-Vers%C3%A3o-Global-Space/product-reviews/B07Y9ZHLXW/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
           'https://www.amazon.com.br/Xiaomi-Redmi-Note-128GB-4GB/product-reviews/B088HJ3FCX/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
           'https://www.amazon.com.br/Smartphone-Xiaomi-Redmi-Note-9S/product-reviews/B085S4DSZH/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
           'https://www.amazon.com.br/Celular-Xiaomi-Redmi-Vers%C3%A3o-Global/product-reviews/B089WCSTLY/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
           'https://www.amazon.com.br/Smartphone-Xiaomi-Redmi-Android-Octa-Core/product-reviews/B08697N43N/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews', 
           'https://www.amazon.com.br/Redmi-Note-128gb-6gb-RAM/product-reviews/B085S4M9Z7/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
           'https://www.amazon.com.br/Redmi-Note-9S-Aurora-128GB/product-reviews/B086CXR646/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_reviews&filterByStar=four_star&pageNumber=1',
           'https://www.amazon.com.br/Smartphone-Xiaomi-Redmi-Note-Camera/product-reviews/B07Z5BBG56/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews']
           # coleta as avaliações
n = len(lst_url)
for i in range(4,n):
    url =  lst_url[i]
    df = get_data(url)
    
    path = 'outputs/dados_avaliacão_consumidor_parte_{}.csv'.format(i+1)
    df.to_csv(path, index=False)

# junta os dados obtidos de diferentes urls
master = []
for i in range(n):
    df = pd.read_csv('outputs/dados_avaliacão_consumidor_parte_{}.csv'.format(i+1))
    master.append(df)
    
df_total = pd.concat(master)
path_total = 'outputs/dados_avaliacão_consumidor_total.csv'
df_total.to_csv(path_total, index=False)

#resumo dos dados
df_resumo = df_total.groupby('Rating').count()
path_resumo = 'outputs/resumo_avaliacão_consumidor.csv'
df_resumo.to_csv(path_resumo, index=False)
