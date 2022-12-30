#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
College Basketball Predictions
@author: brianszekely
"""
import cbb_web_scraper
from os import getcwd
from os.path import join, exists 
import yaml
from tqdm import tqdm
from time import sleep
from pandas import DataFrame
class cbb_regressor():
    def __init__(self):
        print('initialize class cbb_regressor')
        self.all_data = DataFrame()
    def get_teams(self):
        year_list_find = []
        year_list = [2023,2022,2021,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010]
        if exists(join(getcwd(),'year_count.yaml')):
            with open(join(getcwd(),'year_count.yaml')) as file:
                year_counts = yaml.load(file, Loader=yaml.FullLoader)
        else:
            year_counts = {'year':year_list_find}
        if year_counts['year']:
            year_list_check =  year_counts['year']
            year_list_find = year_counts['year']
            year_list = [i for i in year_list if i not in year_list_check]
            print(f'Need data for year: {year_list}')
        if year_list:   
            for year in tqdm(year_list):
                all_teams = cbb_web_scraper.get_teams_year(year)
                team_names = sorted(all_teams)
                final_list = []
                self.year_store = year
                for abv in tqdm(team_names):    
                    # try:
                        print() #tqdm things
                        print(f'current team: {abv}, year: {year}')
                        # https://www.basketball-reference.com/teams/BOS/2023/gamelog/
                        basic = 'https://www.sports-reference.com/cbb/schools/' + abv + '/' + str(self.year_store) + '-gamelogs.html'
                        adv = 'https://www.sports-reference.com/cbb/schools/' + abv + '/' + str(self.year_store) + '-gamelogs-advanced.html'
                        df_inst = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,abv,self.year_store)
                        final_list.append(df_inst)
                    # except:
                    #     print(f'{abv} data are not available')
                        sleep(5) #I get get banned for a small period of time if I do not do this
    def run_analysis(self):
        self.get_teams()
def main():
    cbb_regressor().run_analysis()
if __name__ == '__main__':
    main()