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
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
import xgboost as xgb
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.activations import relu, tanh, linear
from keras_tuner import RandomSearch
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
import os

"""
TODO:
-remove outliers from dataset
-normalize and standardize features
-change random forest to ensemble learning - catboost or xgboost
-correlation or pca analysis?
-create train, valid, and test splits
-check class balance for win and loss
-during the prediction, make the data that team 1 is facing team2
"""
def create_sequential_model(hp, n_features, n_outputs):
    model = Sequential()
    #Add hidden layers
    for i in range(hp.Int('num_layers', 1, 10)):
        if i == 0:
            # First hidden layer needs input shape
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=8, max_value=128, step=8),
                            activation=hp.Choice(f'activation_{i}', values=['relu', 'leaky_relu', 'tanh', 'linear']),
                            kernel_regularizer=l2(hp.Float(f'regularizer_strength_{i}', min_value=1e-1, max_value=1, sampling='log')),
                            input_shape=(n_features,)))
        else:
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=8, max_value=128, step=8),
                            activation=hp.Choice(f'activation_{i}', values=['relu', 'leaky_relu', 'tanh', 'linear']),
                            kernel_regularizer=l2(hp.Float(f'regularizer_strength_{i}', min_value=1e-1, max_value=1, sampling='log'))))
            model.add(BatchNormalization())
            model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.3, max_value=0.6, step=0.1)))
    
    # Output layer
    model.add(Dense(n_outputs, activation='sigmoid'))  # Binary classification
    
    # Compile model
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop']) #, 'sgd'
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=hp.Float('adam_learning_rate', min_value=0.0001, max_value=0.01, sampling='log'))
    else:
        optimizer = RMSprop(learning_rate=hp.Float('rmsprop_learning_rate', min_value=0.0001, max_value=0.01, sampling='log'))
    
    model.compile(optimizer=optimizer,
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

class cbbClass():
    def __init__(self):
        print('instantiate class cbbClass')
        self.all_data = DataFrame()
        # if exists(join(getcwd(),'randomForestModelTuned.joblib')):
        #     self.RandForRegressor=joblib.load("./randomForestModelTuned.joblib")
    def get_teams(self):
        year_list_find = []
        year_list = [2024,2023]#,2022,2021,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010]
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
                all_teams = cbb_web_scraper.get_teams_year(year_list[-1],2024)
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
                if exists(join(getcwd(),'all_data.csv')):
                    self.all_data = read_csv(join(getcwd(),'all_data.csv'))  
                self.all_data = concat([self.all_data, final_data.dropna()])
                if not exists(join(getcwd(),'all_data.csv')):
                    self.all_data.to_csv(join(getcwd(),'all_data.csv'),index=False)
                self.all_data.to_csv(join(getcwd(),'all_data.csv'),index=False)
                year_list_find.append(year)
                print(f'year list after loop: {year_list_find}')
                with open(join(getcwd(),'year_count.yaml'), 'w') as write_file:
                    yaml.dump(year_counts, write_file)
                    print(f'writing {year} to yaml file')
        else:
            self.all_data = read_csv(join(getcwd(),'all_data.csv'))
        print('dataset size: ', np.shape(self.all_data))
        self.all_data = self.all_data.drop_duplicates(keep='last')
        print(f'dataset size after duplicates are dropped: {np.shape(self.all_data)}')
    
    def pca_analysis(self):
        pca = PCA(n_components=0.95) #explain 95% of the variance
        self.x_pca = pca.fit_transform(self.x)

        #Visualize PCA components
        plt.figure()
        plt.figure(figsize=(8, 6))
        plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio of Principal Components')
        plt.savefig('pca_components.png',dpi=400)
        plt.close()

    def convert_to_float(self):
        for col in self.all_data.columns:
            self.all_data[col].replace('', np.nan, inplace=True)
            self.all_data[col] = self.all_data[col].astype(float)
        self.all_data.dropna(inplace=True)

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
        #self.y = np.delete(self.y, np.where(np.isnan(self.x_no_corr)), axis=0)
        #self.x_no_corr = self.x_no_corr.dropna()
        self.y = self.all_data['game_result'].astype(int)
        result_counts = self.all_data['game_result'].value_counts()
        #plot the counts
        plt.figure(figsize=(8, 6))
        result_counts.plot(kind='bar')
        plt.xlabel('Game Result')
        plt.ylabel('Count')
        plt.title('Count of Labels')
        plt.savefig('class_label_count.png',dpi=400)

        #onehot encode
        self.y = to_categorical(self.y)
        self.x = self.all_data.drop(columns=['game_result'])
        #correlational analysis and outlier removal
        self.pre_process_corr_out_remove()
        #pca 
        self.pca_analysis()
        # #Dropna and remove all data from subsequent y data
        # real_values = ~self.x_no_corr.isna().any(axis=1)
        # self.x_no_corr.dropna(inplace=True)
        # self.y = self.y.loc[real_values]

        #75/15/10 split
        #Split data into training and the rest (75% training, 25% temporary)
        self.x_train, x_temp, self.y_train, y_temp = train_test_split(self.x_no_corr, self.y, train_size=0.75, random_state=42)
        #Split the rest into validation and test data (60% validation, 40% test)
        validation_ratio = 0.15 / (1 - 0.75)  # Adjust ratio for the remaining part
        self.x_validation, self.x_test, self.y_validation, self.y_test = train_test_split(x_temp, y_temp, train_size=validation_ratio, random_state=42)

    def pre_process_corr_out_remove(self):
        # Remove features with a correlation coef greater than 0.90
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.90)]
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
        cols = self.x_no_corr.columns
        print(f'Columns dropped  >= 0.90: {to_drop}')
        #Drop samples that are outliers 
        print(f'old feature dataframe shape before outlier removal: {self.x_no_corr.shape}')
        for col_name in cols:
            Q1 = np.percentile(self.x_no_corr[col_name], 5)
            Q3 = np.percentile(self.x_no_corr[col_name], 95)
            IQR = Q3 - Q1
            upper = np.where(self.x_no_corr[col_name] >= (Q3+2.5*IQR)) #1.5 is the standard, use two to see if more data helps improve model performance
            lower = np.where(self.x_no_corr[col_name] <= (Q1-2.5*IQR)) 
            self.x_no_corr.drop(upper[0], inplace = True)
            self.x_no_corr.drop(lower[0], inplace = True)
            self.y = np.delete(self.y, upper[0], axis=0)
            self.y = np.delete(self.y, lower[0], axis=0)
            # self.y.drop(upper[0], inplace = True)
            # self.y.drop(lower[0], inplace = True)
            if 'level_0' in self.x_no_corr.columns:
                self.x_no_corr.drop(columns=['level_0'],inplace = True)
            self.x_no_corr.reset_index(inplace = True)
            # self.y.reset_index(inplace = True, drop=True)
        self.x_no_corr.drop(columns=['level_0','index'],inplace = True)
        print(f'new feature dataframe shape after outlier removal: {self.x_no_corr.shape}')
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")    
        plt.tight_layout()
        plt.savefig('correlations_class.png',dpi=250)
        plt.close()

    # def random_forest_analysis(self):
    #     if argv[1] == 'tune':
    #         #RANDOM FOREST REGRESSOR
    #         RandForclass = RandomForestClassifier()
    #         #Use the number of features as a stopping criterion for depth
    #         rows, cols = self.x_train.shape
    #         cols = int(cols / 2.5) #try to avoid overfitting on depth
    #         #square root of the total number of features is a good limit
    #         # cols = int(np.sqrt(cols))
    #         #parameters to tune
    #         #increasing min_samples_leaf, this will reduce overfitting
    #         Rand_perm = {
    #             'criterion' : ["gini","entropy"], #absolute_error - takes forever to run
    #             'n_estimators': range(300,500,100),
    #             # 'min_samples_split': np.arange(2, 5, 1, dtype=int),
    #             'max_features' : [1, 'sqrt', 'log2'],
    #             'max_depth': np.arange(2,cols,1),
    #             'min_samples_leaf': np.arange(2,4,1)
    #             }
    #         clf_rand = GridSearchCV(RandForclass, Rand_perm, 
    #                             scoring=['accuracy','f1'],
    #                             cv=5,
    #                            refit='accuracy',
    #                            verbose=4, 
    #                            n_jobs=-1)
    #         search_rand = clf_rand.fit(self.x_train,self.y_train)
    #         #Write fitted and tuned model to file
    #         # with open('randomForestModelTuned.pkl','wb') as f:
    #         #     pickle.dump(search_rand,f)
    #         joblib.dump(search_rand, "./classifierModelTuned.joblib", compress=9)
    #         print('RandomForestClassifier - best params: ',search_rand.best_params_)
    #         self.RandForclass = search_rand
    #         prediction = self.RandForclass.predict(self.x_test)
    #         print(confusion_matrix(self.y_test, prediction))# Display accuracy score
    #         print(f'Model accuracy: {accuracy_score(self.y_test, prediction)}')# Display F1 score
    #         # print(f1_score(self.y_test, prediction))
    #     else:
    #         print('Load tuned Random Forest Classifier')
    #         # load RandomForestModel
    #         self.RandForclass=joblib.load("./classifierModelTuned.joblib")
    #         prediction = self.RandForclass.predict(self.x_test)
    #         print(confusion_matrix(self.y_test, prediction))# Display accuracy score
    #         print(f'Model accuracy: {accuracy_score(self.y_test, prediction)}')# Display F1 score
    #         # print(f1_score(self.y_test, prediction))
    #     y_proba = self.RandForclass.predict_proba(self.x_test)[:, 1]
    #     fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
    #     plt.plot(fpr, tpr)
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC Curve')
    #     plt.savefig('ROC_curve_class.png',dpi=300)
    
    def xgboost_analysis(self):
        if not os.path.exists('classifierModelTuned_xgb.joblib'):
            y_train_combined = np.concatenate([self.y_train, self.y_validation], axis=0)
            x_train_combined = concat([self.x_train, self.x_validation], axis=0)
            if argv[1] == 'tune':
                # XGBoost Classifier
                xgb_class = xgb.XGBClassifier()

                # Parameters to tune
                params = {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': range(100, 300, 100),
                    'max_depth': range(2, 4, 2),
                    'min_child_weight': [1, 5],
                    'gamma': [0, 0.2],
                    'subsample': [0.6, 1.0],
                    'colsample_bytree': [0.6, 1.0],
                    'reg_alpha': [0, 0.01],
                    'reg_lambda': [0, 0.01],
                    'scale_pos_weight': [1, 3]
                }

                clf_xgb = GridSearchCV(xgb_class, params,
                                    scoring=['accuracy'],
                                    cv=5,
                                    refit='accuracy',
                                    verbose=4)
                search_xgb = clf_xgb.fit(x_train_combined, y_train_combined)

                # Write fitted and tuned model to file
                joblib.dump(search_xgb, "./classifierModelTuned_xgb.joblib", compress=9)
                print('XGBoost Classifier - best params: ', search_xgb.best_params_)
                self.xgb_class = search_xgb
                prediction = self.xgb_class.predict(self.x_test)
                print('Confusion Matrix: \n',confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(prediction, axis=1)))  # Display accuracy score
                print(f'Model accuracy on test data:: {accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(prediction, axis=1))}')  # Display F1 score

            else:
                print('Load tuned XGBoost Classifier')
                # load XGBoost Model
                self.xgb_class = joblib.load("./classifierModelTuned_xgb.joblib")
                prediction = self.xgb_class.predict(self.x_test)
                print('Confusion Matrix on test data: \n',confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(prediction, axis=1)))  # Display accuracy score
                print(f'Model accuracy on test data: {accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(prediction, axis=1))}')  # Display F1 score

            y_proba = self.xgb_class.predict_proba(self.x_test)
            fpr, tpr, thresholds = roc_curve(np.argmax(self.y_test, axis=1), np.argmax(y_proba, axis=1))
            plt.plot(fpr, tpr)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.savefig('ROC_curve_class.png', dpi=300)
        else:
            self.xgb_class = joblib.load("./classifierModelTuned_xgb.joblib")


    def deep_learn_analysis(self):
        if not os.path.exists('binary_keras_deep.h5'):
            tuner = RandomSearch(
                    lambda hp: create_sequential_model(hp, self.x_train.shape[1], 2),
                    objective='val_loss', #val_loss
                    max_trials=10,
                    directory=f'cbb_sequential_hp',
                    project_name='sequential_hyperparameter_tuning',
                )

            early_stopping = EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            tuner.search(x=self.x_train, y=self.y_train,
                        epochs=200,
                        validation_data=(self.x_validation, self.y_validation),
                        callbacks=[early_stopping, reduce_lr])

            # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model = tuner.get_best_models(num_models=1)[0]

            # Fit tuned model
            loss_final = float(100)
            for i in tqdm(range(15)):
                best_model.fit(self.x_train, self.y_train,
                            epochs=200, 
                            validation_data=(self.x_validation, self.y_validation),
                            callbacks=[early_stopping, reduce_lr])
                loss, acc = best_model.evaluate(self.x_test, self.y_test)
                if loss < loss_final:
                    self.final_model_deep = best_model
            loss, acc = self.final_model_deep.evaluate(self.x_test, self.y_test)
            print(f'Final model test loss {loss} and accuracy {acc}')
            self.final_model_deep.save('binary_keras_deep.h5')
        else:
            self.final_model_deep = load_model('binary_keras_deep.h5')

    def predict_two_teams(self):
        teams_sports_ref = read_csv('teams_sports_ref_format.csv')
        while True:
            try:
                team_1 = input('team_1: ')
                if team_1 == 'exit':
                    break
                team_2 = input('team_2: ')
                #Game location
                game_loc_team1 = int(input(f'{team_1} : home = 0, away = 1, neutral = 2: '))
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
                year = 2024
                # sleep(4)
                basic = 'https://www.sports-reference.com/cbb/schools/' + team_1.lower() + '/' + str(year) + '-gamelogs.html'
                adv = 'https://www.sports-reference.com/cbb/schools/' + team_1.lower() + '/' + str(year) + '-gamelogs-advanced.html'
                team_1_df2023 = cbb_web_scraper.html_to_df_web_scrape_cbb(basic,adv,team_1.lower(),year)
                sleep(4) #I get get banned for a small period of time if I do not do this
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
                ma_range = np.arange(2,5,1) #2 was the most correct value for mean and 8 was the best for the median; chose 9 for tiebreaking
                # team_1_count = 0
                # team_2_count = 0
                # team_1_count_mean = 0
                # team_2_count_mean = 0
                team_1_ma_win = []
                team_2_ma_win = []
                # team_1_ma_loss = []
                # team_2_ma = []
                #get latest SRS value
                team_1_srs = cbb_web_scraper.get_latest_srs(team_1)
                team_2_srs = cbb_web_scraper.get_latest_srs(team_2)

                #rolling average predictions
                for ma in tqdm(ma_range):
                    # data1_median = team_1_df2023.rolling(ma).median()
                    # data1_median['game_loc'] = game_loc_team1
                    # data2_median = team_2_df2023.rolling(ma).median()
                    # data2_median['game_loc'] = game_loc_team2
                    # data1_mean_old = team_1_df2023.rolling(ma).mean()
                    # data2_mean_old = team_2_df2023.rolling(ma).mean()
                    # TEAM 1
                    data1_mean = team_1_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    data1_mean['game_loc'] = game_loc_team1
                    data2_mean = team_2_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    data2_mean['game_loc'] = game_loc_team2
                    # team_1_predict_median = self.RandForclass.predict(data1_median.iloc[-1:])
                    # team_2_predict_median = self.RandForclass.predict(data2_median.iloc[-1:])
                    #Here replace opponent metrics with the features of the second team
                    for col in data1_mean.columns:
                        if "opp" in col:
                            if col == 'opp_trb':
                                # new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], 'opp_trb'] = data2_mean.loc[data2_mean.index[-1], 'total_board']
                            else:
                                new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], col] = data2_mean.loc[data2_mean.index[-1], new_col]
                    #get latest SRS value
                    data1_mean.loc[data1_mean.index[-1], 'simple_rating_system'] = team_1_srs
                    data2_mean.loc[data2_mean.index[-1], 'simple_rating_system'] = team_2_srs 
                    # data1_mean['simple_rating_system'].iloc[-1] = cbb_web_scraper.get_latest_srs(team_1)
                    outcome = self.xgb_class.predict_proba(data1_mean.iloc[-1:])
                    outcome_deep = self.final_model_deep.predict(data1_mean.iloc[-1:])
                    #TEAM 2
                    # data1_mean_change = team_1_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    # data1_mean_change['game_loc'] = game_loc_team1
                    # data2_mean_change = team_2_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    # data2_mean_change['game_loc'] = game_loc_team2
                    # team_1_predict_median = self.RandForclass.predict(data1_median.iloc[-1:])
                    # team_2_predict_median = self.RandForclass.predict(data2_median.iloc[-1:])
                    #Here replace opponent metrics with the features of the second team
                    # for col in team_2_df2023.columns:
                    #     if "opp" in col:
                    #         if col == 'opp_trb':
                    #             # new_col = col.replace("opp_", "")
                    #             data2_mean_change.loc[data2_mean_change.index[-1], 'opp_trb'] = data1_mean_change.loc[data1_mean_change.index[-1], 'total_board']
                    #         else:
                    #             new_col = col.replace("opp_", "")
                    #             data2_mean_change.loc[data2_mean_change.index[-1], col] = data1_mean_change.loc[data1_mean_change.index[-1], new_col]
                    # team_2_predict_mean = self.RandForclass.predict_proba(data2_mean_change.iloc[-1:])
                    team_1_ma_win.append(outcome[0][1])
                    team_1_ma_win.append(outcome_deep[0][1])
                    team_2_ma_win.append(outcome[0][0])
                    team_2_ma_win.append(outcome_deep[0][0])
                #quantile predictions - both play at their bests
                qt_best_team_1, qt_best_team_2 = [], []
                for ma in tqdm(ma_range):
                    # data1_median = team_1_df2023.rolling(ma).median()
                    # data1_median['game_loc'] = game_loc_team1
                    # data2_median = team_2_df2023.rolling(ma).median()
                    # data2_median['game_loc'] = game_loc_team2
                    # data1_mean_old = team_1_df2023.rolling(ma).mean()
                    # data2_mean_old = team_2_df2023.rolling(ma).mean()
                    # TEAM 1
                    data1_mean = team_1_df2023.rolling(window=ma).quantile(0.75).iloc[-1:]
                    data1_mean['game_loc'] = game_loc_team1
                    data2_mean = team_2_df2023.rolling(window=ma).quantile(0.75).iloc[-1:]
                    data2_mean['game_loc'] = game_loc_team2
                    # team_1_predict_median = self.RandForclass.predict(data1_median.iloc[-1:])
                    # team_2_predict_median = self.RandForclass.predict(data2_median.iloc[-1:])
                    #Here replace opponent metrics with the features of the second team
                    for col in data1_mean.columns:
                        if "opp" in col:
                            if col == 'opp_trb':
                                # new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], 'opp_trb'] = data2_mean.loc[data2_mean.index[-1], 'total_board']
                            else:
                                new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], col] = data2_mean.loc[data2_mean.index[-1], new_col]
                    #get latest SRS value
                    data1_mean.loc[data1_mean.index[-1], 'simple_rating_system'] = team_1_srs
                    data2_mean.loc[data2_mean.index[-1], 'simple_rating_system'] = team_2_srs 
                    # data1_mean['simple_rating_system'].iloc[-1] = cbb_web_scraper.get_latest_srs(team_1)
                    outcome = self.xgb_class.predict_proba(data1_mean.iloc[-1:])
                    outcome_deep = self.final_model_deep.predict(data1_mean.iloc[-1:])

                    qt_best_team_1.append(outcome[0][1])
                    qt_best_team_1.append(outcome_deep[0][1])
                    qt_best_team_2.append(outcome[0][0])
                    qt_best_team_2.append(outcome_deep[0][0])

                #quantile predictions - both play at their worsts
                qt_worst_team_1, qt_worst_team_2 = [], []
                for ma in tqdm(ma_range):
                    # data1_median = team_1_df2023.rolling(ma).median()
                    # data1_median['game_loc'] = game_loc_team1
                    # data2_median = team_2_df2023.rolling(ma).median()
                    # data2_median['game_loc'] = game_loc_team2
                    # data1_mean_old = team_1_df2023.rolling(ma).mean()
                    # data2_mean_old = team_2_df2023.rolling(ma).mean()
                    # TEAM 1
                    data1_mean = team_1_df2023.rolling(window=ma).quantile(0.25).iloc[-1:]
                    data1_mean['game_loc'] = game_loc_team1
                    data2_mean = team_2_df2023.rolling(window=ma).quantile(0.25).iloc[-1:]
                    data2_mean['game_loc'] = game_loc_team2
                    # team_1_predict_median = self.RandForclass.predict(data1_median.iloc[-1:])
                    # team_2_predict_median = self.RandForclass.predict(data2_median.iloc[-1:])
                    #Here replace opponent metrics with the features of the second team
                    for col in data1_mean.columns:
                        if "opp" in col:
                            if col == 'opp_trb':
                                # new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], 'opp_trb'] = data2_mean.loc[data2_mean.index[-1], 'total_board']
                            else:
                                new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], col] = data2_mean.loc[data2_mean.index[-1], new_col]
                    #get latest SRS value
                    data1_mean.loc[data1_mean.index[-1], 'simple_rating_system'] = team_1_srs
                    data2_mean.loc[data2_mean.index[-1], 'simple_rating_system'] = team_2_srs 
                    # data1_mean['simple_rating_system'].iloc[-1] = cbb_web_scraper.get_latest_srs(team_1)
                    outcome = self.xgb_class.predict_proba(data1_mean.iloc[-1:])
                    outcome_deep = self.final_model_deep.predict(data1_mean.iloc[-1:])

                    qt_worst_team_1.append(outcome[0][1])
                    qt_worst_team_1.append(outcome_deep[0][1])
                    qt_worst_team_2.append(outcome[0][0])
                    qt_worst_team_2.append(outcome_deep[0][0])
                # team_2_ma_win = []
                # team_2_ma_loss = []
                # for ma in tqdm(ma_range):
                #     # data1_median = team_1_df2023.rolling(ma).median()
                #     # data1_median['game_loc'] = game_loc_team1
                #     # data2_median = team_2_df2023.rolling(ma).median()
                #     # data2_median['game_loc'] = game_loc_team2
                #     # data1_mean_old = team_1_df2023.rolling(ma).mean()
                #     # data2_mean_old = team_2_df2023.rolling(ma).mean()
                #     # TEAM 1
                #     data1_mean = team_1_df2023.ewm(span=ma,min_periods=ma-1).mean()
                #     data1_mean['game_loc'] = game_loc_team1
                #     data2_mean = team_2_df2023.ewm(span=ma,min_periods=ma-1).mean()
                #     data2_mean['game_loc'] = game_loc_team2
                #     # team_1_predict_median = self.RandForclass.predict(data1_median.iloc[-1:])
                #     # team_2_predict_median = self.RandForclass.predict(data2_median.iloc[-1:])
                #     #Here replace opponent metrics with the features of the second team
                #     for col in data2_mean.columns:
                #         if "opp" in col:
                #             if col == 'opp_trb':
                #                 # new_col = col.replace("opp_", "")
                #                 data2_mean.loc[data2_mean.index[-1], 'opp_trb'] = data1_mean.loc[data1_mean.index[-1], 'total_board']
                #             else:
                #                 new_col = col.replace("opp_", "")
                #                 data2_mean.loc[data2_mean.index[-1], col] = data1_mean.loc[data1_mean.index[-1], new_col]
                #     #get latest SRS value
                #     data1_mean.loc[data1_mean.index[-1], 'simple_rating_system'] = team_1_srs
                #     data2_mean.loc[data2_mean.index[-1], 'simple_rating_system'] = team_2_srs 
                #     # data1_mean['simple_rating_system'].iloc[-1] = cbb_web_scraper.get_latest_srs(team_1)
                #     team_2_predict_mean = self.RandForclass.predict_proba(data2_mean.iloc[-1:])
                #     team_2_ma_win.append(team_2_predict_mean[0][1])
                #     team_2_ma_loss.append(team_2_predict_mean[0][0])
                # # team_2_ma.append(team_2_predict_mean[0][1])
                # team_1_win_proba = round(np.mean(team_1_ma_win),4)*100
                # team_1_loss_proba = round(np.mean(team_1_ma_loss),4)*100
                # team_2_win_proba = round(np.mean(team_2_ma_win),4)*100
                # team_2_loss_proba = round(np.mean(team_2_ma_loss),4)*100
                # team_2_proba = 100 - team_1_proba
                # print(f'{team_2} win probability {round(np.median(team_2_predict_mean),4)*100}%')
                # print(f'{team_2} winning: {np.mean(team_2_ma)}%')
                
                print('===============================================================')
                print(f'{team_1} SRS data: {team_1_srs}')
                print(f'{team_2} SRS data: {team_2_srs}')
                print('===============================================================')
                print(f'{team_1} average win probabilities: {np.median(team_1_ma_win)}')
                print(f'{team_2} average win probabilities: {np.median(team_2_ma_win)}')
                print('===============================================================')
                print(f'{team_1} average win probabilities if they play at their best: {np.median(qt_best_team_1)}')
                print(f'{team_2} average win probabilities if they play at their best: {np.median(qt_best_team_2)}')
                print('===============================================================')
                print(f'{team_1} average win probabilities if they play at their worst: {np.median(qt_worst_team_1)}')
                print(f'{team_2} average win probabilities if they play at their worst: {np.median(qt_worst_team_2)}')
                print('===============================================================')
                # if "tod" in sys.argv[2]:
                #     date_today = str(datetime.now().date()).replace("-", "")
                # elif "tom" in sys.argv[2]:
                #     date_today = str(datetime.now().date() + timedelta(days=1)).replace("-", "")
                # URL = "https://www.espn.com/mens-college-basketball/schedule/_/date/" + date_today #sys argv????
                # print(f'ESPN prediction: {cbb_web_scraper.get_espn(URL,team_1,team_2)}')
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
    
    def feature_importances_xgb(self):
        importances = self.xgb_class.best_estimator_.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(10,8))
        plt.title('Feature Importances XGBoost - Classifier')
        plt.barh(range(len(indices)), importances[indices], color='k', align='center')
        plt.yticks(range(len(indices)), [self.x_test.columns[i] for i in indices])
        plt.xlabel('Relative Importance - explained variance')
        plt.tight_layout()
        plt.savefig('feature_importance_xgb_classifier.png',dpi=300)
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
        # self.random_forest_analysis()
        self.deep_learn_analysis()
        self.xgboost_analysis()
        self.predict_two_teams()
        # self.feature_importances_xgb()
def main():
    cbbClass().run_analysis()
if __name__ == '__main__':
    main()