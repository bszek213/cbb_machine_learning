#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
College Basketball Predictions based on EWM over last 3 years
@author: brianszekely
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
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
import seaborn as sns
from sys import argv
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import cross_val_score, KFold
import pickle
import joblib
import sys
import os
class cbbRegressorEwm():
    def __init__(self):
        print('initialize class cbbRegressorEwm')
        self.all_data = DataFrame()
        
    def get_teams(self):
        year_list_find = []
        year_list = [2023,2022,2021]
        self.years = year_list
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
        #Collect data for all years for each team so that you can take an EWM of the data
        if year_list:
            all_teams = cbb_web_scraper.get_teams_year(year_list[-1],year_list[0])
            team_names = sorted(all_teams)
            for abv in tqdm(team_names):
                final_list = []
                try:
                    for year in year_list:
                        print() #tqdm things
                        print(f'current team: {abv}, year: {year}')
                        basic = 'https://www.sports-reference.com/cbb/schools/' + abv + '/' + str(year) + '-gamelogs.html'
                        adv = 'https://www.sports-reference.com/cbb/schools/' + abv + '/' + str(year) + '-gamelogs-advanced.html'
                        df_inst = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,abv,year)
                        df_inst['pts'].replace('', np.nan, inplace=True)
                        df_inst.dropna(inplace=True)
                        final_list.append(df_inst)
                        sleep(5) #I get get banned for a small period of time if I do not do this
                    final_data = concat(final_list)
                    final_data.drop(columns='game_result',inplace=True)
                    #Perform EWM on all data except for pts
                    for col in final_data.columns:
                        if col != 'pts':
                            final_data[col] = final_data[col].ewm(span=3,ignore_na=True).mean() #this is the arbitrary part: 3 games since the best outcomes from the old analysis were between 2-4 games
                    final_data['team'] = abv #I love how in python the data size is just "figured" out...this is such bad coding
                    #Save data to file
                    if exists(join(getcwd(),'all_data_regressor.csv')):
                        self.all_data = read_csv(join(getcwd(),'all_data_regressor.csv'))  
                    self.all_data = concat([self.all_data, final_data.dropna()])
                    if not exists(join(getcwd(),'all_data_regressor.csv')):
                        self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'),index=False)
                    self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'),index=False)
                except Exception as e:
                    print(e)
                    print(f'{abv} data are not available')
            #Write years to file
            with open(join(getcwd(),'year_count.yaml'), 'w') as write_file:
                year_counts = {'year':year_list}
                yaml.dump(year_counts, write_file)
                print(f'writing {year} to yaml file')
        else:
            self.all_data = read_csv(join(getcwd(),'all_data_regressor.csv'))
    def convert_to_float(self):
        for col in self.all_data.columns:
            if 'team' not in col:
                self.all_data[col] = self.all_data[col].astype(float)
    def pre_process(self):
        # Remove features with a correlation coef greater than 0.85
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.85)]
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
        self.drop_cols.append('opp_pts') #may use this in the future as a feature
        cols = self.x_no_corr.columns
        print(f'Columns dropped: {self.drop_cols}')
        #Drop samples that are outliers 
        print(f'old feature dataframe shape before outlier removal: {self.x_no_corr.shape}')
        for col_name in cols:
            Q1 = np.percentile(self.x_no_corr[col_name], 25)
            Q3 = np.percentile(self.x_no_corr[col_name], 75)
            IQR = Q3 - Q1
            upper = np.where(self.x_no_corr[col_name] >= (Q3+3.0*IQR)) #1.5 is the standard, use two to see if more data helps improve model performance
            lower = np.where(self.x_no_corr[col_name] <= (Q1-3.0*IQR)) 
            self.x_no_corr.drop(upper[0], inplace = True)
            self.x_no_corr.drop(lower[0], inplace = True)
            self.y.drop(upper[0], inplace = True)
            self.y.drop(lower[0], inplace = True)
            if 'level_0' in self.x_no_corr.columns:
                self.x_no_corr.drop(columns=['level_0'],inplace = True)
            self.x_no_corr.reset_index(inplace = True)
            self.y.reset_index(inplace = True, drop=True)
        self.x_no_corr.drop(columns=['level_0','index'],inplace = True)
        print(f'new feature dataframe shape after outlier removal: {self.x_no_corr.shape}')
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(30,30))
        sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")    
        plt.tight_layout()
        plt.savefig('correlations.png',dpi=300)
        plt.close()
    def split(self):
        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        self.convert_to_float()
        self.y = self.all_data['pts']
        self.x = self.all_data.drop(columns=['pts','team','opp_pts'])
        self.pre_process()
        #Dropna and remove all data from subsequent y data
        real_values = ~self.x_no_corr.isna().any(axis=1)
        self.x_no_corr.dropna(inplace=True)
        self.y = self.y.loc[real_values]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_no_corr, self.y, train_size=0.8)
    def random_forest_analysis(self):
        if argv[1] == 'tune':
            #RANDOM FOREST REGRESSOR
            RandForclass = RandomForestRegressor()
            #Use the number of features as a stopping criterion for depth
            rows, cols = self.x_train.shape
            #square root of the total number of features is a good limit
            # cols = int(np.sqrt(cols))
            #parameters to tune
            #increasing min_samples_leaf, this will reduce overfitting
            Rand_perm = {
                'criterion' : ["squared_error", "poisson"], #absolute_error - takes forever to run
                'n_estimators': range(300,500,100),
                # 'min_samples_split': np.arange(2, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2'],
                'max_depth': np.arange(2,cols,1),
                'min_samples_leaf': np.arange(1,3,1)
                }
            clf_rand = GridSearchCV(RandForclass, Rand_perm, 
                                scoring=['neg_root_mean_squared_error','explained_variance'],
                                cv=10,
                                refit='neg_root_mean_squared_error',verbose=4, n_jobs=-1)
            #save
            search_rand = clf_rand.fit(self.x_train,self.y_train)
            #Write fitted and tuned model to file
            joblib.dump(search_rand, "./randomForestModelTuned.joblib", compress=9)
            print('RandomForestRegressor - best params: ',search_rand.best_params_)
        else:
            print('Load tuned Random Forest Regressor')
            self.RandForRegressor=joblib.load("./randomForestModelTuned.joblib")
            print(f'Current RandomForestRegressor Parameters: {self.RandForRegressor.best_params_}')
            print('RMSE: ',mean_squared_error(self.RandForRegressor.predict(self.x_test),self.y_test,squared=False))
            print('R2 score: ',r2_score(self.RandForRegressor.predict(self.x_test),self.y_test))
    def predict_two_teams(self):
        while True:
            try:
                team_1 = input('team_1: ')
                if team_1 == 'exit':
                    break
                team_2 = input('team_2: ')
                #Team 1 data - I hate how I am gathering this
                final_list = []
                year_list = [2023,2022]
                for year in year_list:
                    basic = 'https://www.sports-reference.com/cbb/schools/' + team_1 + '/' + str(year) + '-gamelogs.html'
                    adv = 'https://www.sports-reference.com/cbb/schools/' + team_1 + '/' + str(year) + '-gamelogs-advanced.html'
                    df_inst = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,team_1,year)
                    df_inst['pts'].replace('', np.nan, inplace=True)
                    df_inst.dropna(inplace=True)
                    final_list.append(df_inst)
                    sleep(5)
                team_1_df = concat(final_list)
                for col in team_1_df.columns:
                        if col != 'pts':
                            team_1_df[col] = team_1_df[col].ewm(span=3,ignore_na=True).mean()
                #Team 2 data - I hate how I am gathering this
                final_list = []
                for year in self.years:
                    basic = 'https://www.sports-reference.com/cbb/schools/' + team_1 + '/' + str(year) + '-gamelogs.html'
                    adv = 'https://www.sports-reference.com/cbb/schools/' + team_1 + '/' + str(year) + '-gamelogs-advanced.html'
                    df_inst = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,team_2,year)
                    df_inst['pts'].replace('', np.nan, inplace=True)
                    df_inst.dropna(inplace=True)
                    final_list.append(df_inst)
                    sleep(5)
                team_2_df = concat(final_list)
                for col in team_2_df.columns:
                        if col != 'pts':
                            team_2_df[col] = team_2_df[col].ewm(span=3,ignore_na=True).mean()
                #Drop the correlated features
                team_1_df.drop(columns=self.drop_cols, inplace=True)
                team_2_df.drop(columns=self.drop_cols, inplace=True)
                team_1_df.drop(columns=['game_result','pts'],inplace=True)
                team_2_df.drop(columns=['game_result','pts'],inplace=True)
                #Predict on the final value in dataframe columns
                team_1_predict_mean = self.RandForRegressor.predict(team_1_df.iloc[-1:])
                team_2_predict_mean = self.RandForRegressor.predict(team_2_df.iloc[-1:])
                print('===============================================================')
                print(f'Outcomes with EWM')
                print(f'{team_1}: {team_1_predict_mean[0]}')
                print(f'{team_2}: {team_2_predict_mean[0]}')
                if team_1_predict_mean[0] > team_2_predict_mean[0]:
                    print(f'{team_1} wins')
                else:
                    print(f'{team_2} wins')
                print('===============================================================')
                # team_1_df = self.all_data[self.all_data['team'].str.contains(team_1)]
                # team_2_df = self.all_data[self.all_data['team'].str.contains(team_2)]
            except Exception as e:
                print(f'The error: {e}')
    def feature_importances_random_forest(self):
        importances = self.RandForRegressor.best_estimator_.feature_importances_
        indices = np.argsort(importances)
        plt.figure()
        plt.title('Feature Importances Random Forest')
        plt.barh(range(len(indices)), importances[indices], color='k', align='center')
        plt.yticks(range(len(indices)), [self.x_test.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_random_forest.png',dpi=300)
    def run_analysis(self):
        self.get_teams()
        self.split()
        self.random_forest_analysis()
        self.predict_two_teams()
        self.feature_importances_random_forest()
def main():
    cbbRegressorEwm().run_analysis()
if __name__ == '__main__':
    main()