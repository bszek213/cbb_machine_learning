#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
College Basketball Predictions via classification and probability with ESPN 
@author: brianszekely
"""
import cbb_web_scraper
from os import getcwd
from os.path import join, exists 
import yaml
from tqdm import tqdm
from time import sleep
from pandas import DataFrame, concat, read_csv, isnull
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
import joblib
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from difflib import get_close_matches
import sys
from datetime import datetime, timedelta
from sklearn.metrics import roc_curve
import seaborn as sns
"""
TODO: change the labels to be a 2x1 array of team_1 = 0, team_2 = 1.
Use the opponent features for team_2 and then the normal features for team_1.
Then I can compare the probability of team_1 beating team_2
"""
class cbbClass():
    def __init__(self):
        print('instantiate class cbbClass')
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
    def convert_to_float(self):
        for col in self.all_data.columns:
            self.all_data[col].replace('', np.nan, inplace=True)
            self.all_data[col] = self.all_data[col].astype(float)
    def delete_opp(self):
        """
        Drop any opponent data, as it may not be helpful when coming to prediction. Hard to estimate with running average
        """
        for col in self.all_data.columns:
            if 'opp' in col:
                self.all_data.drop(columns=col,inplace=True)
    def split(self):
        # self.delete_opp()
        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        self.convert_to_float()
        self.y = self.all_data['game_result'].astype(int)
        self.x = self.all_data.drop(columns=['game_result'])
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
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.90)]
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
        cols = self.x_no_corr.columns
        print(f'Columns dropped  >= 0.90: {to_drop}')
        #Drop samples that are outliers 
        print(f'old feature dataframe shape before outlier removal: {self.x_no_corr.shape}')
        for col_name in cols:
            Q1 = np.percentile(self.x_no_corr[col_name], 25)
            Q3 = np.percentile(self.x_no_corr[col_name], 75)
            IQR = Q3 - Q1
            upper = np.where(self.x_no_corr[col_name] >= (Q3+2.5*IQR)) #1.5 is the standard, use two to see if more data helps improve model performance
            lower = np.where(self.x_no_corr[col_name] <= (Q1-2.5*IQR)) 
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
        plt.figure(figsize=(20,20))
        sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")    
        plt.tight_layout()
        plt.savefig('correlations_class.png',dpi=250)
        plt.close()
    def random_forest_analysis(self):
        if argv[1] == 'tune':
            #RANDOM FOREST REGRESSOR
            RandForclass = RandomForestClassifier()
            #Use the number of features as a stopping criterion for depth
            rows, cols = self.x_train.shape
            cols = int(cols / 2.5) #try to avoid overfitting on depth
            #square root of the total number of features is a good limit
            # cols = int(np.sqrt(cols))
            #parameters to tune
            #increasing min_samples_leaf, this will reduce overfitting
            Rand_perm = {
                'criterion' : ["gini","entropy"], #absolute_error - takes forever to run
                'n_estimators': range(300,500,100),
                # 'min_samples_split': np.arange(2, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2'],
                'max_depth': np.arange(2,cols,1),
                'min_samples_leaf': np.arange(1,3,1)
                }
            clf_rand = GridSearchCV(RandForclass, Rand_perm, 
                                scoring=['accuracy'],
                                cv=10,
                               refit='accuracy',verbose=4, n_jobs=-1)
            search_rand = clf_rand.fit(self.x_train,self.y_train)
            #Write fitted and tuned model to file
            # with open('randomForestModelTuned.pkl','wb') as f:
            #     pickle.dump(search_rand,f)
            joblib.dump(search_rand, "./classifierModelTuned.joblib", compress=9)
            print('RandomForestClassifier - best params: ',search_rand.best_params_)
            self.RandForclass = search_rand
            prediction = self.RandForclass.predict(self.x_test)
            print(confusion_matrix(self.y_test, prediction))# Display accuracy score
            print(f'Model accuracy: {accuracy_score(self.y_test, prediction)}')# Display F1 score
            # print(f1_score(self.y_test, prediction))
        else:
            print('Load tuned Random Forest Classifier')
            # load RandomForestModel
            self.RandForclass=joblib.load("./classifierModelTuned.joblib")
            prediction = self.RandForclass.predict(self.x_test)
            print(confusion_matrix(self.y_test, prediction))# Display accuracy score
            print(f'Model accuracy: {accuracy_score(self.y_test, prediction)}')# Display F1 score
            # print(f1_score(self.y_test, prediction))
        y_proba = self.RandForclass.predict_proba(self.x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig('ROC_curve_class.png',dpi=300)
    def predict_two_teams(self):
        teams_sports_ref = read_csv('teams_sports_ref_format.csv')
        while True:
            try:
                team_1 = input('team_1: ')
                if team_1 == 'exit':
                    break
                team_2 = input('team_2: ')
                #Game location
                game_loc_team1 = int(input(f'{team_1} : #home = 0, away = 1, N = 2: '))
                if game_loc_team1 == 0:
                    game_loc_team2 = 1
                elif game_loc_team1 == 1:
                    game_loc_team2 = 0
                elif game_loc_team1 == 2:
                    game_loc_team2 = 2
                #Check to see if the team was spelled right
                team_1  = get_close_matches(team_1,teams_sports_ref['teams'].tolist(),n=1)[0]
                team_2  = get_close_matches(team_2,teams_sports_ref['teams'].tolist(),n=1)[0]
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
                team_1_df2023.replace('', np.nan, inplace=True)
                team_1_df2023.dropna(inplace=True)
                team_2_df2023['pts'].replace('', np.nan, inplace=True)
                team_2_df2023.replace('', np.nan, inplace=True)
                team_2_df2023.dropna(inplace=True)
                #Remove pts and game result
                # for col in team_1_df2023.columns:
                #     if 'opp' in col:
                #         team_1_df2023.drop(columns=col,inplace=True)
                # for col in team_2_df2023.columns:
                #     if 'opp' in col:
                #         team_2_df2023.drop(columns=col,inplace=True)
                team_1_df2023.drop(columns=['game_result'],inplace=True)
                team_2_df2023.drop(columns=['game_result'],inplace=True)
                #Drop the correlated features
                team_1_df2023.drop(columns=self.drop_cols, inplace=True)
                team_2_df2023.drop(columns=self.drop_cols, inplace=True)
                ma_range = np.arange(2,7,1) #2 was the most correct value for mean and 8 was the best for the median; chose 9 for tiebreaking
                team_1_count = 0
                team_2_count = 0
                team_1_count_mean = 0
                team_2_count_mean = 0
                team_1_ma_win = []
                team_1_ma_loss = []
                team_2_ma = []
                for ma in tqdm(ma_range):
                    data1_median = team_1_df2023.rolling(ma).median()
                    data1_median['game_loc'] = game_loc_team1
                    data2_median = team_2_df2023.rolling(ma).median()
                    data2_median['game_loc'] = game_loc_team2
                    # data1_mean_old = team_1_df2023.rolling(ma).mean()
                    # data2_mean_old = team_2_df2023.rolling(ma).mean()
                    data1_mean = team_1_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    data1_mean['game_loc'] = game_loc_team1
                    data2_mean = team_2_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    data2_mean['game_loc'] = game_loc_team2
                    # team_1_predict_median = self.RandForclass.predict(data1_median.iloc[-1:])
                    # team_2_predict_median = self.RandForclass.predict(data2_median.iloc[-1:])
                    #Here replace opponent metrics with the features of the second team
                    for col in team_1_df2023.columns:
                        if "opp" in col:
                            if col == 'opp_trb':
                                # new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], 'opp_trb'] = data2_mean.loc[data2_mean.index[-1], 'total_board']
                                # data1_mean['opp_trb'].iloc[-1] = data2_mean['total_board'].iloc[-1]
                            else:
                                new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], col] = data2_mean.loc[data2_mean.index[-1], new_col]
                                # data1_mean[col].iloc[-1] = data2_mean[new_col].iloc[-1]
                    team_1_predict_mean = self.RandForclass.predict_proba(data1_mean.iloc[-1:])
                    # team_2_predict_mean = self.RandForclass.predict_proba(data2_mean.iloc[-1:])
                    # both = self.RandForclass.predict_proba(concat([data1_mean.iloc[-1:], data2_mean.iloc[-1:]]))
                    team_1_ma_win.append(team_1_predict_mean[0][1])
                    team_1_ma_loss.append(team_1_predict_mean[0][0])
                # team_2_ma.append(team_2_predict_mean[0][1])
                print('===============================================================')
                print(f'{team_1} win probability {np.mean(team_1_ma_win)}%')
                print(f'{team_1} loss probability {np.mean(team_1_ma_loss)}%')
                # print(f'{team_2} winning: {np.mean(team_2_ma)}%')
                print('===============================================================')
                if np.mean(team_1_ma_win) > np.mean(team_1_ma_loss):
                    print(f'{team_1} wins')
                else:
                    print(f'{team_2} wins')
                print('===============================================================')
                if "tod" in sys.argv[2]:
                    date_today = str(datetime.now().date()).replace("-", "")
                elif "tom" in sys.argv[2]:
                    date_today = str(datetime.now().date() + timedelta(days=1)).replace("-", "")
                URL = "https://www.espn.com/mens-college-basketball/schedule/_/date/" + date_today #sys argv????
                print(f'ESPN prediction: {cbb_web_scraper.get_espn(URL,team_1,team_2)}')
                print('===============================================================')
            except Exception as e:
                print(f'The error: {e}')
    def feature_importances_random_forest(self):
        importances = self.RandForclass.best_estimator_.feature_importances_
        indices = np.argsort(importances)
        plt.figure()
        plt.title('Feature Importances Random Forest - Classifier')
        plt.barh(range(len(indices)), importances[indices], color='k', align='center')
        plt.yticks(range(len(indices)), [self.x_test.columns[i] for i in indices])
        plt.xlabel('Relative Importance - explained variance')
        plt.tight_layout()
        plt.savefig('feature_importance_random_forest_classifier.png',dpi=300)
        # importances = self.RandForclass.best_estimator_.feature_importances_
        # indices = np.argsort(importances)
        # feature_names = [self.x_test.columns[i] for i in indices]
        # plt.figure()
        # sns.set_style("whitegrid")
        # sns.barplot(x=importances[indices], y=feature_names, color='black', orient='h')
        # plt.title('Feature Importances Random Forest - Classifier')
        # plt.xlabel('Relative Importance - explained variance')
        # plt.tight_layout()
        # plt.savefig('feature_importance_random_forest_classifier.png',dpi=300)
    def run_analysis(self):
        self.get_teams()
        self.split()
        self.random_forest_analysis()
        self.predict_two_teams()
        self.feature_importances_random_forest()
def main():
    cbbClass().run_analysis()
if __name__ == '__main__':
    main()