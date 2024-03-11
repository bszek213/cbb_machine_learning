from fuzzywuzzy import process
import pandas as pd

df = pd.read_csv('all_teams_cbb.csv')
teams = pd.read_csv('teams_sports_ref_format.csv')
def find_closest_match(school_name):
    closest_match = process.extractOne(school_name.lower(), teams['teams'])
    return closest_match[0]

df['School'] = df['School'].apply(find_closest_match)

for val in df['School']:
    print(val)