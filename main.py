import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from fuzzywuzzy import process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def save_dataframes_to_csv(dataframes_dict, output_dir='output'):
    """
    Save DataFrames from a dictionary to CSV files using their keys as filenames
    
    Args:
        dataframes_dict (dict): Dictionary of {name: DataFrame} pairs
        output_dir (str): Directory to save CSV files (default: 'output')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert spaces to underscores and add .csv extension
    for name, df in dataframes_dict.items():
        try:
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Clean filename
                clean_name = name.replace(' ', '_') + '.csv'
                filepath = os.path.join(output_dir, clean_name)
                
                # Save to CSV
                df.to_csv(filepath, index=False)
                print(f"Saved {name} to {filepath}")
            else:
                print(f"{name} is not a valid DataFrame or is empty")
        except Exception as e:
            print(f"Failed to save {name}: {str(e)}")

# Section I
def installing_data():
    url_dict = {
        'stats_standard': 'https://fbref.com/en/comps/9/stats/Premier-League-Stats',
        'stats_keeper': 'https://fbref.com/en/comps/9/keepers/Premier-League-Stats',
        'stats_shooting': 'https://fbref.com/en/comps/9/shooting/Premier-League-Stats',
        'stats_passing': 'https://fbref.com/en/comps/9/passing/Premier-League-Stats',
        'stats_gca': 'https://fbref.com/en/comps/9/gca/Premier-League-Stats',
        'stats_defense': 'https://fbref.com/en/comps/9/defense/Premier-League-Stats',
        'stats_possession': 'https://fbref.com/en/comps/9/possession/Premier-League-Stats',
        'stats_misc': 'https://fbref.com/en/comps/9/misc/Premier-League-Stats'
    }

    target = [
        'player', 'nationality', 'team', 'position', 'age', 'games', 'games_starts', 'minutes', 'goals',
        'assists', 'cards_yellow', 'cards_red', 'xg', 'xg_assist', 'progressive_carries', 'progressive_passes',
        'progressive_passes_received', 'goals_per90', 'assists_per90', 'xg_per90', 'xg_assist_per90',
        'gk_goals_against_per90', 'gk_save_pct', 'gk_clean_sheets_pct', 'gk_pens_save_pct',
        'shots_on_target_pct', 'shots_on_target_per90', 'goals_per_shot', 'average_shot_distance',
        'passes_completed', 'passes_pct', 'passes_total_distance', 'passes_pct_short', 'passes_pct_medium',
        'passes_pct_long', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area',
        'crosses_into_penalty_area', 'progressive_passes', 'sca', 'sca_per90', 'gca', 'gca_per90',
        'tackles', 'tackles_won', 'challenges', 'challenges_lost', 'blocks', 'blocked_shots',
        'blocked_passes', 'interceptions', 
        'touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area',
        'take_ons', 'take_ons_won_pct', 'take_ons_tackled_pct',
        'carries', 'carries_progressive_distance', 'progressive_carries', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 
        'passes_received', 'progressive_passes_received',
        'fouls', 'fouled', 'offsides', 'crosses', 'ball_recoveries', 'aerials_won', 'aerials_lost',
        'aerials_won_pct'
    ]
    #print (len(target))
    target_name = ['Player', 'Nation', 'Team', 'Position', 'Age', 'Match played', 'Starts','Minutes',
    'Goals', 'Assists','Yellow cards', 'Red cards','xG', 'xAG','PrgC', 'PrgP',
    'PrgR','Goals per 90', 'Assists per 90','xG per 90', 'xAG per 90',
    'GA90','Save%','CS%','Pen Save%',
    'SoT%','SoT per 90','G per Sh','Average dist',
    'Cmp','Cmp%','TotDist','Cmp%_S','Cmp%_M','Cmp%_L','KP', 'one_third', 'PPA','CrsPA', 'PrgPasses',
    'SCA', 'SCA90','GCA', 'GCA90',
    'Tkl','TklW','Att','Lost','Blocks', 'Block shot', 'Pass', 'Int',
    'Touches', 'Def Pen', 'Def 3rd','Mid 3rd', 'Att 3rd', 'Att Pen',
    'Possession_Att','Possession_Succ%','Possession_Tkld%',
    'Carries', 'Possession_ProDist','Possession_ProgC', 'Possession_one_third', 'Possession_CPA', 'Possession_Mis',
    'Possession_Dis','Rec', 'PrgR',
    'Fls', 'Fld', 'Off', 'Crs','Recov','Won','Lost', 'Won%']
    #print(len(target_name))
    target_name_dict = dict(zip(target, target_name))
    stat_dfs = {}

    for stat_group, url in url_dict.items():
        try:
            # Fetch HTML
            response = requests.get(url)
            print(response)
            response.raise_for_status()
            html_text = response.text
            html_text = html_text.replace('<!--', '')
            html_text = html_text.replace('-->', '')
            time.sleep(1)  # Be polite and avoid overwhelming the server
            soup = BeautifulSoup(html_text, 'html.parser')
            table = soup.find('table', id=stat_group)
            if not table:
                print(f"Table {stat_group} not found")
                stat_dfs[stat_group] = pd.DataFrame()
                continue

            # Extract headers from thead
            headers = []
            data_list = {}
            for tr in table.select('thead'):
                for th in tr.find_all('th'):
                    data_stat = th.get('data-stat')
                    if data_stat in target:
                        headers.append(data_stat)
                        data_list[data_stat] = []
            # Process rows in tbody
            rows = []

            for tr in table.select('tbody'):
                # Skip subheader rows (rows with th elements that aren't player names)

                row_data = []
                for td in tr.find_all('td'):
                    data_stat = td.get('data-stat').strip()
                    if data_stat in headers:
                        if data_stat == 'minutes':
                            mins = int(td.get_text(strip = True).replace(',',''))    
                            data_list[data_stat].append(mins or "N/a")
                        else:
                            data_list[data_stat].append(td.get_text(strip=True) or "N/a")
                if row_data:
                    rows.append(row_data)
            # Create DataFrame
            #print(rows)
            df = pd.DataFrame(data_list)

            
            stat_dfs[stat_group] = df
            print(f"Processed {stat_group} | Shape: {df.shape}")

        except Exception as e:
            print(f"Error in {stat_group}: {str(e)}")
            stat_dfs[stat_group] = pd.DataFrame()

    # Access DataFrames: stat_dfs['stats_standard'], etc.
    #print_stat_dataframes(stat_dfs)
    #save_dataframes_to_csv(stat_dfs, output_dir='output')

    # Initialize merged DataFrame with standard stats
    merged_df = stat_dfs['stats_standard'].copy()

    # Define key columns for merging
    merge_keys = ['player', 'team']

    # Iterate through other stat groups
    for stat_group in url_dict.keys():
        if stat_group == 'stats_standard':
            continue  # Already used as base
        
        df_to_merge = stat_dfs[stat_group]
        
        # Check if merge keys exist
        if not all(col in df_to_merge.columns for col in merge_keys):
            print(f"Skipping {stat_group} (missing player/team columns)")
            continue
        
        # Drop columns already present (except merge keys)
        cols_to_merge = [col for col in df_to_merge.columns if col not in merged_df.columns or col in merge_keys]
        
        # Merge
        merged_df = merged_df.merge(
            df_to_merge[cols_to_merge],
            on=merge_keys,
            how='left'
        )

    # Replace NaN with 'N/a'
    merged_df = merged_df.fillna('N/a')
    #Filter players played more than 90 mins
    merged_df = merged_df[merged_df['minutes'] >= 90]
    merged_df = merged_df[target]

    #sort name
    merged_df = merged_df.sort_values(by = ['player'])
    merged_df.rename(columns=target_name_dict, inplace=True)
    #Save merged DataFrame to CSV
    merged_df.to_csv('results.csv', index=False)


# Section II
def find_top_3_greatest_and_lowest():
    merged_df = pd.read_csv('results.csv')
#merged_df.set_index('Player')
    target_name = merged_df.columns.tolist()

    merged_df = merged_df.replace('N/a', 0) 

    for i in merged_df['Age']:
        cell = str(i)
        temp = int(cell[0:2])* 365 + int(cell[3:])
        i = temp  
    # convert Age to days for comparing
    # Convert columns to numeric where applicable
    for col in merged_df.columns[4:]:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
    print(merged_df['Player'])
    #merged_df.set_index('Player', inplace = True)
    top3_dataframe = pd.DataFrame(columns = merged_df.columns)

    results = []
    stat_columns = [col for col in merged_df.columns[4:]]
    for stat in stat_columns:
        # Skip if all values are zero (no meaningful data)
        if (merged_df[stat] == 0).all():
            continue
        
        # Get top 3 highest and lowest players
        top_highest = merged_df[['Player', stat]].nlargest(3, stat)
        top_lowest = merged_df[['Player', stat]].nsmallest(3, stat)
        
        # Format results
        highest_entries = [f"{row['Player']}: {row[stat]}" for _, row in top_highest.iterrows()]
        lowest_entries = [f"{row['Player']}: {row[stat]}" for _, row in top_lowest.iterrows()]
        
        results.append(
            f"Statistic: {stat}\n"
            f"Highest:\n" + "\n".join(highest_entries) + "\n"
            f"Lowest:\n" + "\n".join(lowest_entries) + "\n"
        )
        str_temp = "---------------------\n".join(results)
        print(str_temp)
        with open('top_3.txt', 'w', encoding = 'utf-8') as f:
            f.write(str_temp)
          
def find_median_and_mean_and_std():
    merged_df = pd.read_csv('results.csv')
    #merged_df.set_index('Player')
    target_name = merged_df.columns.tolist()
    stat_columns = merged_df.columns[4:]

    merged_df[stat_columns] = merged_df[stat_columns].replace('N/a', 0)
    age_row = []
    for i in merged_df['Age']:
        cell = str(i)
        temp = int(cell[0:2])* 365 + int(cell[3:])
        age_row.append(temp)
    merged_df['Age'] = age_row
    # convert Age to days for comparing
    # Convert columns to numeric where applicable
    for col in stat_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
        
    non_stat_columns = ['Player', 'Nation', 'Team', 'Position']
    stat_columns = [col for col in merged_df.columns if col not in non_stat_columns]

    # Initialize lists to store results
    rows = []
    index_names = ['all']

    # Calculate statistics for all players
    medians = merged_df[stat_columns].median()
    means = merged_df[stat_columns].mean()
    stds = merged_df[stat_columns].std()

    # Create row for league-wide stats
    all_stats = {}
    
    for col in stat_columns:
        all_stats[f'Median of {col}'] = medians[col]
        all_stats[f'Mean of {col}'] = means[col]
        all_stats[f'Std of {col}'] = stds[col]
    rows.append(all_stats)

    # Calculate statistics for each team
    teams = merged_df['Team'].unique()
    for team in teams:
        # Filter a dataframe only have players in the chossed team
        team_df = merged_df[merged_df['Team'] == team]
        team_medians = team_df[stat_columns].median()
        team_means = team_df[stat_columns].mean()
        team_stds = team_df[stat_columns].std()
        
        team_stats = {}
        for col in stat_columns:
            team_stats[f'Median of {col}'] = team_medians[col]
            team_stats[f'Mean of {col}'] = team_means[col]
            team_stats[f'Std of {col}'] = team_stds[col]
        rows.append(team_stats)
        index_names.append(team)

    # Create DataFrame for results
    results_df = pd.DataFrame(rows, index=index_names)

    # Save to results2.csv
    results_df.to_csv('results2.csv', index = index_names)

def plot_histogram():

    merged_df = pd.read_csv('results.csv')
    #merged_df.set_index('Player')
    target_name = merged_df.columns.tolist()

    non_stat_columns = ['Player', 'Nation', 'Team', 'Position']
    stat_columns = [col for col in merged_df.columns if col not in non_stat_columns]


    age_row = []
    for i in merged_df['Age']:
        cell = str(i)
        temp = float(cell[0:2]) + float(int(cell[3:])/365)
        age_row.append(temp)
    merged_df['Age'] = age_row
    #print (merged_df['Age'])
    # Convert columns to numeric where applicable
    for col in stat_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)

    output_dir = 'Histogram'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_dir = 'All'
    if not os.path.exists(all_dir):
        os.makedirs(os.path.join(output_dir, all_dir))

    team_dir = 'Teams'
    if not os.path.exists(os.path.join(output_dir, team_dir)):
        os.makedirs(os.path.join(output_dir, team_dir))

    teams = merged_df['Team'].unique()

    for col in stat_columns:
        plt.figure(figsize = (12, 10))

        plt.hist(merged_df[col].dropna(), bins = 40, color = 'blue', alpha = 0.7, edgecolor = 'green', linewidth = 0.5)

        plt.title(f'Distribution of {col} for all players')
        #Lable the x axis
        plt.xlabel(col)
        #Lable the y axis
        plt.ylabel('Frequency')

        plt.grid(True, alpha = 0.3)

        plt.savefig(os.path.join(output_dir, 'All', f'{col}_all_players.png'))
        plt.close()

    for team in teams:
        temp_dir = os.path.join(output_dir, team_dir, team)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok = True)
        team_df = merged_df[merged_df['Team'] == team]

        for col in stat_columns:
            plt.figure(figsize = (12, 10))

            plt.hist(team_df[col].dropna(), bins = 40, color = 'blue', alpha = 0.7, edgecolor = 'green', linewidth = 0.5)

            plt.title(f'Distribution of {col} for {team}')
            #Lable the x axis
            plt.xlabel(col)
            #Lable the y axis
            plt.ylabel('Frequency')

            plt.grid(True, alpha = 0.3)
            team = team.replace(" ", "_")
            plt.savefig(os.path.join(temp_dir, f'{col}_{team}.png'))
            print(f'Saved histogram for {team}')
            plt.close()

def best_team():
    df = pd.read_csv('results2.csv', header = 0)

    #Requirements:
        #Identify the team with the highest score for each statistic
        #Base on your alalysis, which team do you think is performing the best?

    #Solution:
        # Because football is a team sport, so the score of team is the mean of the players' score.
        # First: Comparing the mean of each statistic for teams
        # Second: Comparing the median of each statistic for teams
        # Third: Comparing the standard deviation of each statistic for teams
    df.columns = ['Team'] + list(df.columns[1:])

    team_df = df[df[df.columns[0]] != 'all']

    columns = [col for col in df.columns if col != 'Team']

    mean_cols = [col for col in columns if col.startswith('Mean of')]
    median_cols = [col for col in columns if col.startswith('Median of')]
    std_cols = [col for col in columns if col.startswith('Std of')]

    highest_mean = {}
    highest_median = {}
    highest_std = {}

    def get_stats(col_name):
        return col_name.split(' of ', 1)[1] # get the statistic name

    # find the highest mean team

    for col in mean_cols:
        stat = get_stats(col)
        if stat in ["GA90", "Age", "Red Cards", "Yellow Cards"]:
            team = team_df.loc[team_df[col].idxmin(), "Team"]
            value = team_df[col].min()
        else:
            team = team_df.loc[team_df[col].idxmax(), "Team"]
            value = team_df[col].max()
        highest_mean[stat] = (team, value)
    # find the highest median team

    for col in median_cols:
        stat = get_stats(col)
        if stat == "GA90":
            team = team_df.loc[team_df[col].idxmin(), "Team"]
            value = team_df[col].min()
        else:
            team = team_df.loc[team_df[col].idxmax(), "Team"]
            value = team_df[col].max()
        highest_median[stat] = (team, value)

    # find the highest std team

    for col in std_cols:
        stat = get_stats(col)
        if stat == "GA90":
            team = team_df.loc[team_df[col].idxmin(), "Team"]
            value = team_df[col].min()
        else:
            team = team_df.loc[team_df[col].idxmax(), "Team"]
            value = team_df[col].max()
        highest_std[stat] = (team, value)
    top_sort = {}
    for stat, (team, value) in highest_mean.items():
        if team not in top_sort:
            top_sort[team] = 0
        top_sort[team] += value

    for team, count in top_sort.items():
        print(f'{team}: {count}')

    best_team = max(top_sort, key=top_sort.get)
    # Save the result of highest mean, median and std team to a text file
    with open('Best statistics.txt', 'w') as file:
        file.write('Best mean:\n')
        for stat, (team, value) in highest_mean.items():
            file.write(f'{stat}: {team} ({value})\n')
        file.write('\nBest median:\n')
        for stat, (team, value) in highest_median.items():
            file.write(f'{stat}: {team} ({value})\n')
        file.write('\nBest std:\n')
        for stat, (team, value) in highest_std.items():
            file.write(f'{stat}: {team} ({value})\n')
        file.write(f'Best team: {best_team}')

    # To identify the best team, we can compare the mean of all statistics for each team
    # and find the team with the total highest mean score.
    print({best_team})
# Section III
def K_means():
    df = pd.read_csv('results.csv')

    # Keep the original dataframe for reference
    df_original = df.copy()

    # Drop non-numeric columns for clustering
    df = df.drop(columns=['Player', 'Nation', 'Team', 'Position'])

    temp_age = []
    for i in range(len(df['Age'])):
        temp = str(df['Age'][i])
        num = int(temp[:2]) + float(temp[3:]) / 365
        temp_age.append(num)
    df['Age'] = temp_age
    df = df.replace('N/a', 0)
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Determine optimal number of clusters using the elbow method
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    # Set optimal number of clusters (assumed k=3 based on typical elbow curve inspection)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Add cluster numbers to the dataframe
    df['Cluster'] = clusters

    # Define key features for each player group
    goalkeeper_features = ['Save%', 'CS%']
    defensive_features = ['Tkl', 'Int', 'Blocks']
    offensive_features = ['Goals', 'Assists', 'xG']

    # Get feature names (excluding 'Cluster' column)
    feature_names = list(df.columns[:-1])

    # Get indices of key features
    gk_indices = [feature_names.index(f) for f in goalkeeper_features if f in feature_names]
    def_indices = [feature_names.index(f) for f in defensive_features if f in feature_names]
    off_indices = [feature_names.index(f) for f in offensive_features if f in feature_names]

    # Compute average centroid values for each group of features and assign labels
    centroids = kmeans.cluster_centers_
    cluster_labels = []
    for i in range(optimal_k):
        gk_score = np.mean(centroids[i, gk_indices]) if gk_indices else -np.inf
        def_score = np.mean(centroids[i, def_indices]) if def_indices else -np.inf
        off_score = np.mean(centroids[i, off_indices]) if off_indices else -np.inf
        scores = [gk_score, def_score, off_score]
        max_score_idx = np.argmax(scores)
        if max_score_idx == 0:
            label = 'Goalkeepers'
        elif max_score_idx == 1:
            label = 'Defensive Players'
        else:
            label = 'Offensive Players'
        cluster_labels.append(label)

    # Map cluster numbers to descriptive labels
    cluster_map = {i: label for i, label in enumerate(cluster_labels)}

    # Add labels to the dataframe
    df['Cluster_Label'] = df['Cluster'].map(cluster_map)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster_Label'] = df['Cluster_Label']

    # Plot clusters with labels
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Label', data=pca_df, palette='viridis', s=100)
    plt.title('K-means Clustering with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster Type')
    plt.savefig('section_3_kmeans.png')
    plt.close()

    # Print cluster labels for reference
    print("Cluster Labels:")
    for i, label in enumerate(cluster_labels):
        print(f"Cluster {i}: {label}")

    # Optional: Verify with position distribution using original data
    df_original = df_original.iloc[df.index]  # Align indices after any row drops
    df_original['Cluster_Label'] = df['Cluster_Label']
    position_distribution = df_original.groupby('Cluster_Label')['Position'].value_counts()
    print("\nPosition Distribution per Cluster:")
    print(position_distribution)
# Section IV
def transform_money(value_str):
    # Remove the euro symbol
    value_str = value_str.replace('€', '')
    
    # Initialize multiplier
    multiplier = 1
    
    # Check for suffixes
    if value_str.endswith('M'):
        multiplier = 1_000_000
        value_str = value_str[:-1]  # Remove 'M'
    elif value_str.endswith('K'):
        multiplier = 1_000
        value_str = value_str[:-1]  # Remove 'K'
    
    # Convert to float and multiply
    try:
        value = float(value_str) * multiplier
        return int(value)
    except ValueError:
        return None  # Return None for invalid input
    
def preprocessing(results_df, value_table):


    results_df['Minutes'] = pd.to_numeric(results_df['Minutes'], errors = 'coerce')

    value_forming = []
    for i in value_table['estimated_value']:
        value_forming.append(transform_money(i))
    value_table ['estimated_value'] = value_forming
    pass

def get_best_match(name, choices, threshold = 74):
    match, score = process.extractOne(name, choices)
    return match if score >= threshold else None

def merging(filtered_results, value_table):
    value_names = value_table['player_name'].tolist()

    filtered_results['Matched_name'] = filtered_results['Player'].apply(
        lambda name: get_best_match(name, value_names)
    )

    filtered_results = filtered_results.dropna(subset = ['Matched_name'])

    # merge databases

    merged_df = filtered_results.merge(
        value_table[['player_name', 'estimated_value']],
        left_on = 'Matched_name',
        right_on = 'player_name',
        how = 'left'
    )
    return merged_df
    pass


def estimating_transfer_value():
    url = "https://www.footballtransfers.com/us/values/actions/most-valuable-football-players/overview"
    header = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            'Referer': "https://www.footballtransfers.com/us/values/players/most-valuable-soccer-players/playing-in-uk-premier-league",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            }

    payload = {
            "orderBy": "estimated_value",
            "orderByDescending": 1,
            "page": 1,
            "pages": 0,
            "pageItems": 25,
            "positionGroupId": 'all',
            "mainPositionId": 'all',
            "playerRoleId": 'all',
            "age": 'all',
            "countryId": 'all',
            "tournamentId": 31
        }




    df = pd.DataFrame()
    for i in range(1, 23):
        payload['page'] = i
        res = requests.post(url, headers = header, data = payload)
        
        content = res.json()
        page_df = pd.DataFrame(content['records'])

        df = pd.concat([df, page_df], axis = 0, ignore_index = True)
    # Delete leftovre columes
    df = df.drop(labels = 'id', axis = 1)
    new_df = df['player_name']
    new_df = pd.concat([new_df, df['team_name']], axis = 1, ignore_index= False)
    new_df = pd.concat([new_df, df['estimated_value']], axis = 1, ignore_index = False)

    new_df.to_csv('transfer_table.csv', index = False)
    value_table = pd.read_csv('transfer_table.csv')
    results_df = pd.read_csv('results.csv')
    filltered_results = results_df.copy()

    preprocessing(results_df, value_table)

    merged_df = merging(filltered_results, value_table)

    #print(merged_df)

    #merged_df.to_csv('merged.csv', index = False)

    merged_df = merged_df.replace('N/a', 0)

    target = 'predict_value'

    features = ['Goals', 'xG', 'Goals per 90', 'Assists', 'xAG', 'Carries', 'Touches',
                'Tkl', 'Blocks', 'Cmp', 'Cmp%', 'one_third',
                'Save%', 'CS%', 'GA90']

    X = merged_df[features]
    y = merged_df['estimated_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    # Scale features

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Train model

    model = RandomForestRegressor(n_estimators= 100, random_state = 1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    import matplotlib.pyplot as plt
    import seaborn as sns

    importances = model.feature_importances_
    sns.barplot(x = importances, y = features)
    plt.xlabel('Importances')
    plt.ylabel('Feature')
    plt.title('Value Estimating Main Standard')
    plt.savefig('Value_estimating.png')
    plt.show()

#installing_data()

#find_top_3_greatest_and_lowest()

#find_median_and_mean_and_std()

#plot_histogram()

#best_team()

#K_means()

#estimating_transfer_value()
