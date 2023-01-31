#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
html parse code - college basketball
@author: brianszekely
"""
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
from numpy import nan
from time import sleep
from os.path import join, exists
from os import getcwd
from urllib import request
from urllib.request import Request, urlopen
from pandas import read_csv
from numpy import where
from re import search
def get_teams_year(year_min,year_max):
    #Try to redo this when 429 is not an issue
    # URL = 'https://www.sports-reference.com/cbb/schools/'
    # hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    # req = Request(URL,headers=hdr)
    # html = request.urlopen(req)
    # soup = BeautifulSoup(html, "html.parser")
    # table = soup.find(class_="table_container is_setup")
    # print(soup)
    # input()
    #Read in from csv
    teams_save = []
    teams = read_csv('all_teams_cbb.csv')
    teams_with_year = where((teams['From'] <= year_min) & (teams['To'] == year_max))[0]
    for team in teams['School'].iloc[teams_with_year]:
        team = team.replace(' ', '-').lower()
        if '.' in team:
            team = team.replace(".", "")
        if 'the' in team:
            team = team.replace("the-", "")
        if '&' in team:
            team = team.replace("&", "")
        if '(' in team and ')' in team:
            team = team.replace("(", "")
            team = team.replace(")", "")
        if "'" in team:
            team = team.replace("'", "")
        teams_save.append(team)
    return teams_save

def html_to_df_web_scrape_cbb(URL,URL1,team,year):
    #URL = Basic data ; URL1 = Advanced stats
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    req_1 = Request(URL,headers=hdr)
    html_1 = request.urlopen(req_1)
    req_2 = Request(URL1,headers=hdr)
    html_2 = request.urlopen(req_2)
    # while True:4700++6
        # try:
    soup_1 = BeautifulSoup(html_1, "html.parser")
    soup_2 = BeautifulSoup(html_2, "html.parser")
            # page = requests.get(URL)
            # soup = BeautifulSoup(page.content, "html.parser")
            # page1 = requests.get(URL1)
            # soup1 = BeautifulSoup(page1.content, "html.parser")
        #     break
        # except:
        #     print('HTTPSConnectionPool(host="www.sports-reference.com", port=443): Max retries exceeded. Retry in 10 seconds')
        #     sleep(10)
    table = soup_1.find(id="all_sgl-basic")
    table1 = soup_2.find(id="all_sgl-advanced")
    tbody = table.find('tbody')
    tbody1 = table1.find('tbody')
    tr_body = tbody.find_all('tr')
    tr_body1 = tbody1.find_all('tr')
    # game_season = []
    # date_game = []
    # game_location = []
    # opp_id= []
    # BASIC STATS
    game_result= []
    pts= []
    opp_pts= []
    fg= []
    fga= []
    fg_pct= []
    fg3= []
    fg3a= []
    fg3_pct= []
    ft= []
    fta= []
    ft_pct= []
    orb= []
    total_board= []
    ast= []
    stl= []
    blk= []
    tov= []
    pf= []
    opp_fg = []
    opp_fga= []
    opp_fg_pct= []
    opp_fg3= []
    opp_fg3a= []
    opp_fg3_pct= []
    opp_ft= []
    opp_fta= []
    opp_ft_pct= []
    opp_orb= []
    opp_trb= []
    opp_ast= []
    opp_stl= []
    opp_blk= []
    opp_tov= []
    opp_pf= []
    game_loc = []
    #BASIC STATS - change td.get_text() to float(td.get_text())
    for trb in tr_body:
        for td in trb.find_all('td'):
            if td.get('data-stat') == "game_location":
                #home = 0, away = 1, N = 2
                if td.get_text() == 'N':
                    game_loc.append(2)
                elif td.get_text() == '@':
                    game_loc.append(1)
                elif td.get_text() == '':
                    game_loc.append(0)
            if td.get('data-stat') == "game_result":
                if td.get_text() == 'W':
                    game_result.append(1)
                else:
                    game_result.append(0)
            if td.get('data-stat') == "pts":
                pts.append(td.get_text())
            if td.get('data-stat') == "opp_pts":
                opp_pts.append(td.get_text())
            if td.get('data-stat') == "fg":
                fg.append(td.get_text())
            if td.get('data-stat') == "fga":
                fga.append(td.get_text())
            if td.get('data-stat') == "fg_pct":
                fg_pct.append(td.get_text())
            if td.get('data-stat') == "fg3":
                fg3.append(td.get_text())
            if td.get('data-stat') == "fg3a":
                fg3a.append(td.get_text())
            if td.get('data-stat') == "fg3_pct":
                fg3_pct.append(td.get_text())
            if td.get('data-stat') == "ft":
                ft.append(td.get_text())
            if td.get('data-stat') == "fta":
                fta.append(td.get_text())
            if td.get('data-stat') == "ft_pct":
                ft_pct.append(td.get_text())
            if td.get('data-stat') == "orb":
                orb.append(td.get_text())
            if td.get('data-stat') == "trb":
                total_board.append(td.get_text())
            if td.get('data-stat') == "ast":
                ast.append(td.get_text())
            if td.get('data-stat') == "stl":
                stl.append(td.get_text())
            if td.get('data-stat') == "blk":
                blk.append(td.get_text())
            if td.get('data-stat') == "tov":
                tov.append(td.get_text())
            if td.get('data-stat') == "pf":
                pf.append(td.get_text())
            if td.get('data-stat') == "opp_fg":
                opp_fg.append(td.get_text())
            if td.get('data-stat') == "opp_fga":
                opp_fga.append(td.get_text())
            if td.get('data-stat') == "opp_fg_pct":
                opp_fg_pct.append(td.get_text())
            if td.get('data-stat') == "opp_fg3":
                opp_fg3.append(td.get_text())
            if td.get('data-stat') == "opp_fg3a":
                opp_fg3a.append(td.get_text())
            if td.get('data-stat') == "opp_fg3_pct":
                opp_fg3_pct.append(td.get_text())
            if td.get('data-stat') == "opp_ft":
                opp_ft.append(td.get_text())
            if td.get('data-stat') == "opp_fta":
                opp_fta.append(td.get_text())
            if td.get('data-stat') == "opp_ft_pct":
                opp_ft_pct.append(td.get_text())
            if td.get('data-stat') == "opp_orb":
                opp_orb.append(td.get_text())
            if td.get('data-stat') == "opp_trb":
                opp_trb.append(td.get_text())
            if td.get('data-stat') == "opp_ast":
                opp_ast.append(td.get_text())
            if td.get('data-stat') == "opp_stl":
                opp_stl.append(td.get_text())
            if td.get('data-stat') == "opp_blk":
                opp_blk.append(td.get_text())
            if td.get('data-stat') == "opp_tov":
                opp_tov.append(td.get_text())
            if td.get('data-stat') == "opp_pf":
                opp_pf.append(td.get_text())      
    #ADVANCED STATS
    off_rtg = []
    def_rtg = []
    pace = []
    fta_per_fga_pct = []
    fg3a_per_fga_pct = []
    ts_pct = []
    trb_pct = []
    ast_pct = []
    stl_pct = [] 
    blk_pct = []
    efg_pct = []
    tov_pct = []
    orb_pct = []
    ft_rate = []
    opp_efg_pct= []
    opp_tov_pct = []
    drb_pct = []
    opp_ft_rate = []
    for trb in tr_body1:
        for td in trb.find_all('td'):
            if td.get('data-stat') == "off_rtg":
                off_rtg.append(td.get_text())
            if td.get('data-stat') == "def_rtg":
                def_rtg.append(td.get_text())
            if td.get('data-stat') == "pace":
                pace.append(td.get_text())
            if td.get('data-stat') == "fta_per_fga_pct":
                fta_per_fga_pct.append(td.get_text())
            if td.get('data-stat') == "fg3a_per_fga_pct":
                fg3a_per_fga_pct.append(td.get_text())
            if td.get('data-stat') == "ts_pct":
                ts_pct.append(td.get_text())
            if td.get('data-stat') == "trb_pct":
                trb_pct.append(td.get_text())
            if td.get('data-stat') == "ast_pct":
                ast_pct.append(td.get_text())
            if td.get('data-stat') == "stl_pct":
                stl_pct.append(td.get_text())
            if td.get('data-stat') == "blk_pct":
                blk_pct.append(td.get_text())
            if td.get('data-stat') == "efg_pct":
                efg_pct.append(td.get_text())
            if td.get('data-stat') == "tov_pct":
                tov_pct.append(td.get_text())
            if td.get('data-stat') == "orb_pct":
                orb_pct.append(td.get_text())
            if td.get('data-stat') == "ft_rate":
                ft_rate.append(td.get_text())
            if td.get('data-stat') == "opp_efg_pct":
                opp_efg_pct.append(td.get_text())
            if td.get('data-stat') == "opp_tov_pct":
                opp_tov_pct.append(td.get_text())
            if td.get('data-stat') == "drb_pct":
                drb_pct.append(td.get_text())
            if td.get('data-stat') == "opp_ft_rate":
                opp_ft_rate.append(td.get_text())
    return DataFrame(list(zip(game_result,pts,opp_pts,fg,fga,
    fg_pct,fg3,fg3a,fg3_pct,ft,fta,ft_pct,orb,total_board,ast,
    stl,blk,tov,pf,opp_fg,opp_fga,opp_fg_pct,opp_fg3,opp_fg3a,opp_fg3_pct,
    opp_ft,opp_fta,opp_ft_pct,opp_orb,opp_trb,opp_ast,opp_stl,opp_blk,opp_tov,
    opp_pf, off_rtg,def_rtg,pace,fta_per_fga_pct,fg3a_per_fga_pct,ts_pct,
    trb_pct,ast_pct,stl_pct,blk_pct,efg_pct,tov_pct,orb_pct,ft_rate,opp_efg_pct,
    opp_tov_pct,drb_pct,opp_ft_rate,game_loc)),
            columns =['game_result','pts','opp_pts','fg','fga',
            'fg_pct','fg3','fg3a','fg3_pct','ft','fta','ft_pct','orb','total_board','ast',
            'stl','blk','tov','pf','opp_fg','opp_fga','opp_fg_pct','opp_fg3','opp_fg3a','opp_fg3_pct',
            'opp_ft','opp_fta','opp_ft_pct','opp_orb','opp_trb','opp_ast','opp_stl','opp_blk','opp_tov',
            'opp_pf','off_rtg','def_rtg','pace','fta_per_fga_pct','fg3a_per_fga_pct','ts_pct',
            'trb_pct','ast_pct','stl_pct','blk_pct','efg_pct','tov_pct','orb_pct','ft_rate','opp_efg_pct',
            'opp_tov_pct','drb_pct','opp_ft_rate','game_loc'])
def get_espn(URL,team_1,team_2):
    # URL = "https://www.espn.com/mens-college-basketball/schedule/_/date/20230131"
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    req_1 = Request(URL,headers=hdr)
    html_1 = request.urlopen(req_1)
    soup_1 = BeautifulSoup(html_1, "html.parser")
    table = soup_1.find(class_="ResponsiveTable")
    table1 = table.find(class_="Table__Scroller")
    table2 = table.find(class_="Table")
    table3 = table.find(class_="Table__TBODY")
    for td in table3.find_all(class_="Table__TR Table__TR--sm Table__even"):
        #Get team names
        inst = td.find(class_="events__col Table__TD")
        href_team = inst.find(class_="AnchorLink").get("href")
        if team_1 in href_team:
            #Get game link
            inst = td.find(class_="date__col Table__TD")
            href_val = inst.find(class_="AnchorLink").get("href")
            game = "https://www.espn.com" + href_val
            req_second = Request(game,headers=hdr)
            html_second = request.urlopen(req_second)
            soup_second = BeautifulSoup(html_second, "html.parser")
            #Team 1 - left-0 top-0 = Away
            team_1_predict = soup_second.find(class_="matchupPredictor__teamValue matchupPredictor__teamValue--b left-0 top-0 flex items-baseline absolute copy")
            start = '>'
            end = "<div"
            team_1_result = float(search('%s(.*)%s' % (start, end), str(team_1_predict)).group(1))
            #Team 2 - bottom-0 right-0 = Home
            team_2_predict = soup_second.find(class_="matchupPredictor__teamValue matchupPredictor__teamValue--a bottom-0 right-0 flex items-baseline absolute copy")
            start = '>'
            end = "<div"
            team_2_result = float(search('%s(.*)%s' % (start, end), str(team_2_predict)).group(1))
            break
    return {team_1: team_1_result, team_2: team_2_result}
    