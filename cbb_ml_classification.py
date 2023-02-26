# ML classification and prediction probabilities
"""
- Predict like normally with 0 and 1 being the output 
- feature engineer running features based on the top 5 features from the old 
ml algorithm.
- then use the code below to compare two teams' probability of beating each other:
# Load the preprocessed data for the two teams
team1_data = pd.read_csv("team1_data.csv")
team2_data = pd.read_csv("team2_data.csv")

# Combine the data for the two teams
matchup_data = pd.concat([team1_data, team2_data])

# Use the trained model to make predictions
probabilities = trained_model.predict_proba(matchup_data)

# Extract the probability of team 1 winning
team1_win_prob = probabilities[0][0]
print("Probability of team 1 winning: ", team1_win_prob)
"""
import cbb_web_scraper
from os import getcwd
from os.path import join, exists 
import yaml
from tqdm import tqdm
from time import sleep
from pandas import DataFrame, concat, read_csv, isnull
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
import seaborn as sns
from sys import argv
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import cross_val_score, KFold
import pickle
import joblib
import sys
import os
from scipy.stats import variation
from difflib import get_close_matches
from datetime import datetime, timedelta
class cbbClassifier():
    def __init__(self):
        print('initialize class cbb_regressor')
        self.all_data = DataFrame()

    def get_teams(self):
        year_list_find = []
        year_list = [2023,2022,2021,2019,2018,2017,2016,2015,2014,2013,2012] #,2014,2013,2012,2011,2010
        if exists(join(getcwd(),'year_count.yaml')):
            with open(join(getcwd(),'year_count.yaml')) as file:
                year_counts = yaml.load(file, Loader=yaml.FullLoader)
        else:
            year_counts = {'year':year_list_find}
        #Remove any years that have already been collected
        if year_counts['year']:
            year_list_check =  year_counts['year']
            year_list_find = year_counts['year']
            year_list = [i for i in year_list if i not in year_list_check]
            print(f'Need data for year: {year_list}')
        #Collect data per year
        if year_list:   
            for year in tqdm(year_list):
                all_teams = cbb_web_scraper.get_teams_year(year_list[-1],year_list[0])
                team_names = sorted(all_teams)
                final_list = []
                self.year_store = year
                for abv in tqdm(team_names):    
                    try:
                        print() #tqdm things
                        print(f'current team: {abv}, year: {year}')
                        basic = 'https://www.sports-reference.com/cbb/schools/' + abv + '/' + str(self.year_store) + '-gamelogs.html'
                        adv = 'https://www.sports-reference.com/cbb/schools/' + abv + '/' + str(self.year_store) + '-gamelogs-advanced.html'
                        df_inst = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,abv,self.year_store)
                        df_inst['pts'].replace('', np.nan, inplace=True)
                        df_inst.dropna(inplace=True)
                        final_list.append(df_inst)
                    except Exception as e:
                        print(e)
                        print(f'{abv} data are not available')
                    sleep(4) #I get get banned for a small period of time if I do not do this
                final_data = concat(final_list)
                if exists(join(getcwd(),'all_data_regressor.csv')):
                    self.all_data = read_csv(join(getcwd(),'all_data_regressor.csv'))  
                self.all_data = concat([self.all_data, final_data.dropna()])
                if not exists(join(getcwd(),'all_data_regressor.csv')):
                    self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'))
                self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'))
                year_list_find.append(year)
                print(f'year list after loop: {year_list_find}')
                with open(join(getcwd(),'year_count.yaml'), 'w') as write_file:
                    yaml.dump(year_counts, write_file)
                    print(f'writing {year} to yaml file')
        else:
            self.all_data = read_csv(join(getcwd(),'all_data_regressor.csv'))
        print('len data: ', len(self.all_data))
        self.all_data = self.all_data.drop_duplicates(keep='last')
        print(f'length of data after duplicates are dropped: {len(self.all_data)}')
    def create_new_features(self):
        pass
    def run_analysis():
        pass

def main():
    cbbClassifier().run_analysis()
if __name__ == '__main__':
    main()