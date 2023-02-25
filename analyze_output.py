#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze output from machine learning model to determine why the model gets some games wrong
@author: brianszekely
"""
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from numpy import where, mean
from scipy.stats import ttest_ind, pearsonr
from seaborn import regplot
def get_data(path):
    return read_csv(path)
def basic_stats(df):
    df.dropna(inplace=True)
    #get difference in outcomes
    df['Team_1_pt_diff'] = abs(df['Team 1 Score'] - df['Team 1 Score Pred'])
    df['Team_2_pt_diff'] = abs(df['Team 2 Score'] - df['Team 2 Score Pred'])
    corr_median = where(df['Correct Median'] == 1)[0]
    incorr_median = where(df['Correct Median'] == 0)[0]
    #NO DIFFERENCE BETWEEN TEAM 1 VAR AND TEAM 2 VAR IN THE INCORRECT OUTCOMES
    # plt.bar('team_1_var',df['Team 1 Var'].iloc[incorr])
    # plt.bar('team_2_var',df['Team 2 Var'].iloc[incorr])
    # print(ttest_ind(df['Team 1 Var'].iloc[incorr],df['Team 2 Var'].iloc[incorr]))
    #NO DIFFERENCE BETWEEN TEAM 1 VAR AND TEAM 2 VAR IN THE CORRECT OUTCOMES
    # plt.bar('team_1_var',df['Team 1 Var'].iloc[corr])
    # plt.bar('team_2_var',df['Team 2 Var'].iloc[corr])
    # print(ttest_ind(df['Team 1 Var'].iloc[corr],df['Team 2 Var'].iloc[corr]))
    #LOW CORRELATIONS BETWEEN VARIABILITY AND ESTIMATED : ACTUAL POINT OUTCOMES
    # regplot(data=df,x='Team 1 Var',y='Team_1_pt_diff',scatter=True,fit_reg=True,label='team1')
    # regplot(data=df,x='Team 2 Var',y='Team_2_pt_diff',scatter=True,fit_reg=True,label='team2')
    # print(pearsonr(df['Team_1_pt_diff'],df['Team 1 Var']))
    # print(pearsonr(df['Team_2_pt_diff'],df['Team 2 Var']))
    # plt.legend()
    #NO DIFFERENCE IN VARIABILITY IN GAMES THAT ARE INCORRECTLY PREDICTED AND HAVE A LARGE PT DIFFERENTIAL COMPARED TO THE 
    #TEAM THAT WAS CLOSER TO THE PREDICTED OUTCOME
    #THERE IS A SIGNIFICANT DIFFERENCE BETWEEN THE TEAM THAT IS BIGGER IN DIFFERENCE THAN THE TEAM HAS A SMALLER DIFFERENCE
    #MAY MEAN THAT ONE TEAM IS BEING INCORRECTLY PREDICTED, WHILE THE OTHER TEAM IS ALMOST SPOT ON
    # greater_diff = []
    # lesser_diff = []
    # for i in range(len(incorr_median)):
    #     if df['Team_1_pt_diff'].iloc[i] > df['Team_2_pt_diff'].iloc[i]:
    #         greater_diff.append(df['Team_1_pt_diff'].iloc[i])
    #         lesser_diff.append(df['Team_2_pt_diff'].iloc[i])
    #     else:
    #         greater_diff.append(df['Team_2_pt_diff'].iloc[i])
    #         lesser_diff.append(df['Team_1_pt_diff'].iloc[i])
    # plt.bar('greater_diff',mean(greater_diff)) 
    # plt.bar('lesser_diff',mean(lesser_diff))
    # print(ttest_ind(greater_diff,lesser_diff))
    #NO CORRELATION BETWEEN BEST TWO FEATURES STD AND PTS DIFF BETWEEN TEAMS WITH HIGH DIFF AND LOW DIFF
    greater_diff = []
    greater_var = []
    lesser_diff = []
    lesser_var = []
    for i in range(len(df)):
        if df['Team_1_pt_diff'].iloc[i] > df['Team_2_pt_diff'].iloc[i]:
            # greater_var.append(df['Team 1 Var'].iloc[i])
            # greater_diff.append(df['Team_1_pt_diff'].iloc[i])
            # lesser_diff.append(df['Team_2_pt_diff'].iloc[i])
            # lesser_var.append(df['Team 2 Var'].iloc[i])
            greater_diff.append(df['Team 1 Var'].iloc[i] / df['Team_1_pt_var'].iloc[i])
            lesser_diff.append(df['Team 2 Var'].iloc[i] / df['Team_2_pt_var'].iloc[i])
        else:
            greater_diff.append(df['Team 2 Var'].iloc[i] / df['Team_2_pt_var'].iloc[i])
            lesser_diff.append(df['Team 1 Var'].iloc[i] / df['Team_1_pt_var'].iloc[i])
            # greater_var.append(df['Team 2 Var'].iloc[i])
            # greater_diff.append(df['Team_2_pt_diff'].iloc[i])
            # lesser_diff.append(df['Team_1_pt_diff'].iloc[i])
            # lesser_var.append(df['Team 1 Var'].iloc[i])
    plt.bar('greater_diff',mean(greater_diff)) 
    plt.bar('lesser_diff',mean(lesser_diff))
    print(ttest_ind(greater_diff,lesser_diff))
    # greater_diff_df = DataFrame({'Team 1 Var': greater_var,'Team_1_pt_diff': greater_diff})
    # lesser_diff_df = DataFrame({'Team 2 Var': lesser_var,'Team_2_pt_diff': lesser_diff})
    # regplot(data=greater_diff_df,x='Team 1 Var',y='Team_1_pt_diff',scatter=True,fit_reg=True,label='greater')
    # regplot(data=lesser_diff_df,x='Team 2 Var',y='Team_2_pt_diff',scatter=True,fit_reg=True,label='lesser')
    # plt.legend()
    # print(pearsonr(greater_diff_df['Team_1_pt_diff'],greater_diff_df['Team 1 Var']))
    # print(pearsonr(lesser_diff_df['Team_2_pt_diff'],lesser_diff_df['Team 2 Var']))
    plt.show()
def main():
    df = get_data('test_acc_regression.csv')
    basic_stats(df)
if __name__ == "__main__":
    main()