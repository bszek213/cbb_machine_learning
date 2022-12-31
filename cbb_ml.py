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
from pandas import DataFrame, concat, read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
import seaborn as sns
from sys import argv
class cbb_regressor():
    def __init__(self):
        print('initialize class cbb_regressor')
        self.all_data = DataFrame()
    def get_teams(self):
        year_list_find = []
        year_list = [2023,2022,2021,2019,2018,2017,2016,2015] #,2014,2013,2012,2011,2010
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
                        # https://www.basketball-reference.com/teams/BOS/2023/gamelog/
                        basic = 'https://www.sports-reference.com/cbb/schools/' + abv + '/' + str(self.year_store) + '-gamelogs.html'
                        adv = 'https://www.sports-reference.com/cbb/schools/' + abv + '/' + str(self.year_store) + '-gamelogs-advanced.html'
                        df_inst = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,abv,self.year_store)
                        df_inst['pts'].replace('', np.nan, inplace=True)
                        df_inst.dropna(inplace=True)
                        final_list.append(df_inst)
                    except Exception as e:
                        print(e)
                        print(f'{abv} data are not available')
                    sleep(5) #I get get banned for a small period of time if I do not do this
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
    def split(self):
        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        self.y = self.all_data['pts']
        self.x = self.all_data.drop(columns=['pts','game_result'])
        self.pre_process()
        #Dropna and remove all data from subsequent y data
        real_values = ~self.x_no_corr.isna().any(axis=1)
        self.x_no_corr.dropna(inplace=True)
        self.y = self.y.loc[real_values]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_no_corr, self.y, train_size=0.8)
        
    def pre_process(self):
        # Remove features with a correlation coef greater than 0.85
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.85)]
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
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
    def random_forest_analysis(self):
        if argv[1] == 'tune':
            #RANDOM FOREST
            RandForclass = RandomForestRegressor()
            rows, cols = self.x_train.shape
            Rand_perm = {
                'criterion' : ["squared_error", "absolute_error", "poisson"],
                'n_estimators': range(100,500,100),
                # 'min_samples_split': np.arange(2, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2'],
                'max_depth': np.arange(3,cols,1)
                }
            #['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'matthews_corrcoef', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']
            clf_rand = GridSearchCV(RandForclass, Rand_perm, scoring=['neg_root_mean_squared_error','explained_variance'],
                               refit='neg_root_mean_squared_error',verbose=4, n_jobs=4)
            search_rand = clf_rand.fit(self.x_train,self.y_train)
            #TODO: Write best params to file
            #with open('data.yml', 'w') as outfile:
                # yaml.dump(data, outfile, default_flow_style=False)
            print('RandomForestRegressor - best params: ',search_rand.best_params_)
    def multi_layer_perceptron(self):
        pass
    def keras_regressor_analysis(self):
        pass
    def run_analysis(self):
        self.get_teams()
        self.split()
        self.random_forest_analysis()
def main():
    cbb_regressor().run_analysis()
if __name__ == '__main__':
    main()