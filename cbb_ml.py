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
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import cross_val_score, KFold
import pickle
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
            #['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 
            # average_precision', 'balanced_accuracy', 'completeness_score', 'explained_variance', 
            # 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 
            # 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 
            # 'jaccard_weighted', 'matthews_corrcoef', 'max_error', 'mutual_info_score', 'neg_brier_score',
            # 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 
            # 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 
            # 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 
            # 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']
            clf_rand = GridSearchCV(RandForclass, Rand_perm, 
                                scoring=['neg_root_mean_squared_error','explained_variance'],
                                cv=10,
                               refit='neg_root_mean_squared_error',verbose=4, n_jobs=-1)
            search_rand = clf_rand.fit(self.x_train,self.y_train)# save
            #Write fitted and tuned model to file
            with open('randomForestModelTuned.pkl','wb') as f:
                pickle.dump(search_rand,f)
            print('RandomForestRegressor - best params: ',search_rand.best_params_)
        else:
            print('Load tuned Random Forest Regressor')
            # load f
            with open('randomForestModelTuned.pkl', 'rb') as f:
                self.RandForRegressor = pickle.load(f)
            print(f'Current RandomForestRegressor Parameters: {self.RandForRegressor.best_params_}')
            print('RMSE: ',mean_squared_error(self.RandForRegressor.predict(self.x_test),self.y_test,squared=False))
            print('R2 score: ',r2_score(self.RandForRegressor.predict(self.x_test),self.y_test))
            # self.RandForRegressor = RandomForestRegressor(criterion='squared_error', 
            #                                               max_depth=20,
            #                                               max_features='log2', 
            #                                               n_estimators=300,
            #                                               min_samples_leaf=3)       
    def multi_layer_perceptron(self):
        pass
    def keras_regressor_analysis(self):
        pass
    def predict_two_teams(self):
        while True:
            try:
                team_1 = input('team_1: ')
                if team_1 == 'exit':
                    break
                team_2 = input('team_2: ')
                #2023 data
                year = 2023
                basic = 'https://www.sports-reference.com/cbb/schools/' + team_1.lower() + '/' + str(year) + '-gamelogs.html'
                adv = 'https://www.sports-reference.com/cbb/schools/' + team_1.lower() + '/' + str(year) + '-gamelogs-advanced.html'
                team_1_df2023 = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,team_1.lower(),year)
                basic = 'https://www.sports-reference.com/cbb/schools/' + team_2.lower() + '/' + str(year) + '-gamelogs.html'
                adv = 'https://www.sports-reference.com/cbb/schools/' + team_2.lower() + '/' + str(year) + '-gamelogs-advanced.html'
                team_2_df2023 = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,team_2.lower(),year)
                #Remove empty cells
                team_1_df2023['pts'].replace('', np.nan, inplace=True)
                team_1_df2023.dropna(inplace=True)
                team_2_df2023['pts'].replace('', np.nan, inplace=True)
                team_2_df2023.dropna(inplace=True)
                #Save series of pts for visualizations
                self.pts_team_1 = team_1_df2023['pts'].astype(float)
                self.team_1_name = team_1
                self.pts_team_2 = team_2_df2023['pts'].astype(float)
                self.team_2_name = team_2
                #Remove pts and game result
                team_1_df2023.drop(columns=['game_result','pts'],inplace=True)
                team_2_df2023.drop(columns=['game_result','pts'],inplace=True)
                #Drop the correlated features
                team_1_df2023.drop(columns=self.drop_cols, inplace=True)
                team_2_df2023.drop(columns=self.drop_cols, inplace=True)
                # team_1_df2023.to_csv('team_1.csv')
                # team_2_df2023.to_csv('team_2.csv')
                print(team_1_df2023)
                print(team_2_df2023)
                #Clean up dataframe
                # for col in team_1_df2023.columns:
                #     if 'Unnamed' in col:
                #         team_1_df2023.drop(columns=col,inplace=True)
                # for col in team_2_df2023.columns:
                #     if 'Unnamed' in col:
                #         team_2_df2023.drop(columns=col,inplace=True)
                #Try to find the moving averages that work
                ma_range = np.arange(2,len(team_2_df2023)-2,1)
                team_1_count = 0
                team_2_count = 0
                team_1_count_mean = 0
                team_2_count_mean = 0
                team_1_ma = []
                team_2_ma = []
                team_1_median = []
                team_2_median = []
                num_pts_score_team_1= []
                num_pts_score_team_2 = []
                for ma in tqdm(ma_range):
                    data1 = team_1_df2023.rolling(ma).median()
                    data2 = team_2_df2023.rolling(ma).median()
                    data1_mean = team_1_df2023.rolling(ma).mean()
                    data2_mean = team_2_df2023.rolling(ma).mean()
                    # print(data1.iloc[-1:])
                    # print(data2.iloc[-1:])
                    # print(data1_mean.iloc[-1:])
                    # print(data2_mean.iloc[-1:])
                    team_1_predict = self.RandForRegressor.predict(data1.iloc[-1:])
                    team_2_predict = self.RandForRegressor.predict(data2.iloc[-1:])
                    team_1_predict_mean = self.RandForRegressor.predict(data1_mean.iloc[-1:])
                    team_2_predict_mean = self.RandForRegressor.predict(data2_mean.iloc[-1:])
                    num_pts_score_team_1.append(team_1_predict_mean[0])
                    num_pts_score_team_2.append(team_2_predict_mean[0])
                    num_pts_score_team_1.append(team_1_predict[0])
                    num_pts_score_team_2.append(team_2_predict[0])
                    if team_1_predict > team_2_predict:
                        team_1_count += 1
                        team_1_median.append(ma)
                    if team_1_predict < team_2_predict:
                        team_2_count += 1
                        team_2_median.append(ma)
                    if team_1_predict_mean > team_2_predict_mean:
                        team_1_count_mean += 1
                        team_1_ma.append(ma)
                    if team_1_predict_mean < team_2_predict_mean:
                        team_2_count_mean += 1
                        team_2_ma.append(ma)
                print('===============================================================')
                print(f'Outcomes with a rolling median from 2-{len(team_2_df2023)} games')
                print(f'{team_1}: {team_1_count} | {team_1_median}')
                print(f'{team_2}: {team_2_count} | {team_2_median}')
                print('===============================================================')
                print(f'Outcomes with a mean from 2-{len(team_2_df2023)} games')
                print(f'{team_1}: {team_1_count_mean} | {team_1_ma}')
                print(f'{team_2}: {team_2_count_mean} | {team_2_ma}')
                print('===============================================================')
                print(f'{team_1} number of pts score: {np.mean(num_pts_score_team_1)}')
                print(f'{team_2} number of pts score: {np.mean(num_pts_score_team_2)}')
                print('===============================================================')
                self.visualization(np.mean(num_pts_score_team_1),np.mean(num_pts_score_team_2))
            except Exception as e:
                print(f'The error: {e}')
                print(f'{team_1} or {team_2} data could not be found. check spelling or internet connection')
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
    def visualization(self,pred_1,pred_2):
        games_1 = range(1,len(self.pts_team_1)+1,1)
        games_2 = range(1,len(self.pts_team_2)+1,1)
        team_1_pred = self.team_1_name + " prediction"
        team_2_pred = self.team_2_name + " prediction"
        plt.figure()
        plt.plot(games_1,self.pts_team_1,color='green',label=self.team_1_name)
        plt.plot(games_2,self.pts_team_2,color='blue',label=self.team_2_name)
        plt.scatter(len(self.pts_team_1)+2,pred_1,color='green',label=team_1_pred)
        plt.scatter(len(self.pts_team_2)+2,pred_2,color='blue',label=team_2_pred)
        plt.legend()
        plt.xlabel('Games')
        plt.ylabel('Points')
        plt.tight_layout()
        plt.show()
    def run_analysis(self):
        self.get_teams()
        self.split()
        self.random_forest_analysis()
        self.predict_two_teams()
        self.feature_importances_random_forest()
def main():
    cbb_regressor().run_analysis()
if __name__ == '__main__':
    main()