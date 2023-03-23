#deep learning implementation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
import cbb_web_scraper
from os import getcwd
from os.path import join, exists 
import yaml
from tqdm import tqdm
from time import sleep
from pandas import DataFrame, concat, read_csv, isnull
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sys import argv
import joblib
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from difflib import get_close_matches
# from datetime import datetime, timedelta
# from sklearn.metrics import roc_curve
import seaborn as sns

#TODO: CREATE A FEATURE OF opp_simple_rating_system

class cbbDeep():
    def __init__(self):
        print('instantiate class cbbClass')
        self.all_data = DataFrame()
        # if exists(join(getcwd(),'randomForestModelTuned.joblib')):
        #     self.RandForRegressor=joblib.load("./randomForestModelTuned.joblib")
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
                    self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'),index=False)
                self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'),index=False)
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
        # normalize data
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
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
    def create_model(self, neurons=32, learning_rate=0.001, dropout_rate=0.2, alpha=0.1):
        model = keras.Sequential([
            layers.Dense(neurons, input_shape=(self.x_no_corr.shape[1],)),
            layers.LeakyReLU(alpha=alpha),
            layers.Dropout(dropout_rate),
            layers.Dense(neurons),
            layers.LeakyReLU(alpha=alpha),
            layers.Dropout(dropout_rate),
            layers.Dense(neurons),
            layers.LeakyReLU(alpha=alpha),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    def deep_learn(self):
        if exists('deep_learning.h5'):
            self.model = keras.models.load_model('deep_learning.h5')
        else:
            #best params
            # Best: 0.999925 using {'alpha': 0.1, 'batch_size': 32, 'dropout_rate': 0.2,
            #  'learning_rate': 0.001, 'neurons': 16}
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            self.model = keras.Sequential([
                    layers.Dense(16, input_shape=(self.x_no_corr.shape[1],)),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(16),
                    layers.LeakyReLU(alpha=0.1),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(1, activation='sigmoid')
                ])
            self.model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
            history = self.model.fit(self.x_train, self.y_train, 
                                     epochs=50, batch_size=32,
                                    validation_data=(self.x_test,self.y_test))
                                     #validation_split=0.2)
            # param_grid = {
            #     'neurons': [16, 32, 64],
            #     'learning_rate': [0.01, 0.001, 0.0001],
            #     'dropout_rate': [0.1, 0.2, 0.3],
            #     'alpha': [0.01, 0.1, 0.2],
            #     'batch_size': [16, 32, 64]
            # }
            # param_grid = {
            #     'neurons': [16, 32],
            #     'learning_rate': [0.01, 0.001],
            #     'dropout_rate': [0.2],
            #     'alpha': [0.1],
            #     'batch_size': [32, 64]
            # }
            # model = KerasClassifier(build_fn=self.create_model, 
            #                         epochs=50, batch_size=32, verbose=4)
            # grid = GridSearchCV(estimator=model, 
            #                     param_grid=param_grid,
            #                     cv=3,
            #                     verbose=3)
            # self.grid_result = grid.fit(self.x_train, self.y_train)
            # print("Best: %f using %s" % (self.grid_result.best_score_, self.grid_result.best_params_))
            # self.model = self.grid_result
            # input()
            self.model.save('deep_learning.h5')
            plt.figure()
            plt.plot(history.history['accuracy'], label='training accuracy')
            plt.plot(history.history['val_accuracy'], label='validation accuracy')
            plt.title('Accuracy History')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('Accuracy.png',dpi=300)
            plt.close()

            # plot loss history
            plt.figure()
            plt.plot(history.history['loss'], label='training loss')
            plt.plot(history.history['val_loss'], label='validation loss')
            plt.title('Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('Loss.png',dpi=300)
            plt.close()
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
                sleep(4)
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
                team_1_count = 0
                team_2_count = 0
                team_1_count_mean = 0
                team_2_count_mean = 0
                team_1_ma_win = []
                team_1_ma_loss = []
                team_2_ma = []
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
                    for col in team_1_df2023.columns:
                        if "opp" in col:
                            if col == 'opp_trb':
                                # new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], 'opp_trb'] = data2_mean.loc[data2_mean.index[-1], 'total_board']
                            else:
                                new_col = col.replace("opp_", "")
                                data1_mean.loc[data1_mean.index[-1], col] = data2_mean.loc[data2_mean.index[-1], new_col]
                    #get latest SRS value
                    print(data1_mean.iloc[-1:])
                    data1_mean.loc[data1_mean.index[-1], 'simple_rating_system'] = cbb_web_scraper.get_latest_srs(team_1)
                    print(data1_mean.iloc[-1:])
                    # data1_mean['simple_rating_system'].iloc[-1] = cbb_web_scraper.get_latest_srs(team_1)# float(input(f'input {team_1} current simple rating system value: '))
                    #TEAM 1 Prediction
                    x_new = self.scaler.transform(data1_mean.iloc[-1:])
                    prediction = self.model.predict(x_new)
                    print(f'prediction: {prediction}')
                    probability = prediction[0]
                    if probability > 0.5:
                        team_1_count += 1
                    elif probability < 0.5:
                        team_2_count += 1
                    # team_1_predict_mean = self.RandForclass.predict_proba(data1_mean.iloc[-1:])
                    #TEAM
                    # data1_mean_change = team_1_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    # data1_mean_change['game_loc'] = game_loc_team1
                    # data2_mean_change = team_2_df2023.ewm(span=ma,min_periods=ma-1).mean()
                    # data2_mean_change['game_loc'] = game_loc_team2
                    # x_new = self.scaler.transform(data2_mean_change.iloc[-1:])
                    # prediction = self.model.predict(x_new)
                    # probability = prediction[0]
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
                # team_2_ma.append(team_2_predict_mean[0][1])
                # print('===============================================================')
                # print(f'{team_1} win probability {round(np.mean(team_1_ma_win),4)*100}%')
                # print(f'{team_2} win probability {round(np.median(team_2_predict_mean),4)*100}%')
                # print(f'{team_2} winning: {np.mean(team_2_ma)}%')
                print('===============================================================')
                # if np.mean(team_1_ma_win) > np.mean(team_1_ma_loss):
                #     print(f'{team_1} wins over {team_2}')
                # else:
                #     print(f'{team_2} wins over {team_1}')
                if team_1_count > team_2_count:
                    print(f'{team_1} wins over {team_2}')
                elif team_1_count < team_2_count:
                    print(f'{team_2} wins over {team_1}')
                print('===============================================================')
            except Exception as e:
                    print(f'The error: {e}')
    def run_analysis(self):
        self.get_teams()
        self.split()
        self.deep_learn()
        self.predict_two_teams()
def main():
    cbbDeep().run_analysis()
if __name__ == '__main__':
    main()