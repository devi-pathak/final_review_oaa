#helper.py

import numpy as np
import pandas as pd

def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df

    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]

    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]

    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]

        # ✅ Convert to numeric and sum medals
        temp_df[['Gold', 'Silver', 'Bronze']] = temp_df[['Gold', 'Silver', 'Bronze']].apply(pd.to_numeric,
                                                                                            errors='coerce').fillna(
            0).astype(int)

        x = temp_df.groupby('Sport')[['Gold', 'Silver', 'Bronze']].sum().reset_index()

        # ✅ Remove rows where all medal counts are zero
        x = x[(x[['Gold', 'Silver', 'Bronze']].sum(axis=1) > 0)]

        # If no medals, return an empty DataFrame with column names
        if x.empty:
            return pd.DataFrame(columns=['Sport', 'Gold', 'Silver', 'Bronze', 'Total'])

        # ✅ Add total column
        x['Total'] = x['Gold'] + x['Silver'] + x['Bronze']
        return x


    if flag == 1:
        # Convert medal columns to numeric before summing
        temp_df[['Gold', 'Silver', 'Bronze']] = temp_df[['Gold', 'Silver', 'Bronze']].apply(pd.to_numeric,
                                                                                            errors='coerce').fillna(
            0).astype(int)

        x = temp_df.groupby('Year')[['Gold', 'Silver', 'Bronze']].sum().sort_values('Year').reset_index()
    else:
        temp_df[['Gold', 'Silver', 'Bronze']] = temp_df[['Gold', 'Silver', 'Bronze']].apply(pd.to_numeric,
                                                                                            errors='coerce').fillna(
            0).astype(int)

        x = temp_df.groupby('region')[['Gold', 'Silver', 'Bronze']].sum().sort_values('Gold', ascending=False)

        #x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold',ascending=False).reset_index()

    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    return x


def medal_tally(df):
    medal_tally = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    medal_tally = medal_tally.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()
    medal_tally['total'] = medal_tally['Gold'] + medal_tally['Silver'] + medal_tally['Bronze']

    return medal_tally

def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years,country

def data_over_time(df,col):
    nations_over_time = df.drop_duplicates(['Year',col])['Year'].value_counts().reset_index().sort_values('Year')
    nations_over_time.rename(columns={'Year': 'Edition', 'count':col}, inplace=True)
    return nations_over_time


def most_successful(df, sport):
    # Filter out rows where Medal is missing or empty
    temp_df = df[(df['Medal'].notna()) & (df['Medal'] != "")].copy()

    # If a specific sport is selected, filter by that sport
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    # If no medal data exists, return an empty DataFrame with proper columns
    if temp_df.empty:
        return pd.DataFrame(columns=['Name', 'Medal_Count', 'Sport', 'region'])

    # Aggregate medal counts per athlete
    medal_counts = (
        temp_df.groupby('Name')
        .agg(Medal_Count=('Medal', 'count'))
        .reset_index()
    )

    # (Optional) Filter out any rows with 0 medal count (shouldn't happen now)
    medal_counts = medal_counts[medal_counts['Medal_Count'] > 0]

    # Retrieve athlete details from the filtered data
    athlete_details = (
        temp_df.groupby('Name')
        .agg({'Sport': 'first', 'region': 'first'})
        .reset_index()
    )

    # Merge medal counts with athlete details
    top_athletes = medal_counts.merge(athlete_details, on='Name', how='left')

    # Sort by medal count in descending order and take only available entries (up to 15)
    top_athletes = top_athletes.sort_values(by='Medal_Count', ascending=False).head(15)

    return top_athletes


def yearwise_medal_tally(df, country):
    temp_df = df.dropna(subset=['Medal']).copy()
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df.loc[temp_df['region'] == country, ['Year', 'Medal']]

    if new_df.empty:
        return pd.DataFrame(columns=['Year', 'Medal'])

    final_df = new_df.groupby('Year', as_index=False).agg({'Medal': 'count'})
    return final_df


def country_event_heatmap(df, country):
    temp_df = df.dropna(subset=['Medal']).copy()
    # Drop duplicate medal records
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    # Filter for the selected country
    new_df = temp_df[temp_df['region'] == country]

    # Remove any extra whitespace in the 'Sport' column
    new_df['Sport'] = new_df['Sport'].str.strip()

    # Create pivot table: index is Sport, columns are Year, values are count of Medals
    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count')

    # Fill missing values and convert to integer
    pt = pt.fillna(0).astype(int)

    # Optionally, remove any rows with empty sport names
    pt = pt[pt.index != '']

    return pt


def most_successful_countrywise(df, country):
    # Filter dataset for rows where Medal is not null.
    temp_df = df.dropna(subset=['Medal']).copy()

    # Use 'region' if available; otherwise, fall back to 'NOC'
    if 'region' in temp_df.columns:
        temp_df = temp_df[temp_df['region'] == country]
    else:
        temp_df = temp_df[temp_df['NOC'] == country]

    # If no data is found, return an empty DataFrame with proper columns.
    if temp_df.empty:
        return pd.DataFrame(columns=['Name', 'Medal_Count', 'Sport'])

    # Convert grouping columns to string to avoid issues with categorical types.
    temp_df['Name'] = temp_df['Name'].astype(str)
    temp_df['Sport'] = temp_df['Sport'].astype(str)

    # Now, perform the aggregation.
    top_athletes = (
        temp_df.groupby(['Name', 'Sport'], as_index=False)
        .agg(Medal_Count=('Medal', 'count'))
        .sort_values(by='Medal_Count', ascending=False)
        .head(10)
    )

    return top_athletes


def weight_v_height(df, sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    if sport != 'Overall':
        athlete_df = athlete_df[athlete_df['Sport'] == sport]

    # ✅ Ensure 'Medal' is a categorical column
    if athlete_df['Medal'].dtype.name == 'category':
        athlete_df['Medal'] = athlete_df['Medal'].cat.add_categories(['No Medal'])

    # ✅ Now safely fill missing medals with 'No Medal'
    athlete_df['Medal'].fillna('No Medal', inplace=True)

    return athlete_df


def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final
