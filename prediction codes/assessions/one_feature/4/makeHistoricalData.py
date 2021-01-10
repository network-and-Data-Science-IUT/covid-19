import pandas as pd
import numpy as np
import datetime
import time
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import requests


######################################################################### NEW RANK
# h is the number of days before day (t)
# r indicates how many days after day (t) --> target-day = day(t+r)
# target could be number of deaths or number of confirmed 
def makeHistoricalData(h, r, test_size, target, feature_selection, spatial_mode, target_mode,
                       address, future_features, pivot, end_date):
    ''' in this code when h is 1, it means there is no history and we have just one column for each covariate
    so when h is 0, we put h equal to 1, because when h is 0 that means there no history (as when h is 1) '''
    if h == 0:
        h = 1
    future_mode = False
    # for r >= 28 we add some new features informing about the future
    future_limit = 4 if target_mode == 'weeklyaverage' else 28
    # if r >= future_limit: future_mode = True
    
    only_death_flag = True
    def impute_feature(data,feature):
        data.loc[data[feature]<0,feature]=np.NaN
        value_count=data.groupby('county_fips').count()
        counties_with_all_nulls=value_count[value_count[feature]==0]
        temp=pd.DataFrame(index=data['county_fips'].unique().tolist(),columns=data['date'].unique().tolist())

        for i in data['date'].unique():
            temp[i]=data.loc[data['date']==i,feature].tolist()
        X = np.array(temp)
        imputer = KNNImputer(n_neighbors=5)
        imp=imputer.fit_transform(X)
        imp=pd.DataFrame(imp)
        imp.columns=temp.columns
        imp.index=temp.index
        for i in data['date'].unique():
            data.loc[data['date']==i,feature]=imp[i].tolist()
        if(len(counties_with_all_nulls)>0):
            data.loc[data['county_fips'].isin(counties_with_all_nulls.index),feature]=np.NaN
        return(data)
    def mean_impute_feature(data,feature):
            temp=pd.DataFrame(index=data['date'].unique().tolist(),columns=data['county_fips'].unique().tolist())
            for i in data['county_fips'].unique():
                temp[i]=data.loc[data['county_fips']==i,feature].tolist()
            for i in temp.columns:
                temp.loc[pd.isna(temp[i]),i]=temp[i].mean()
            for i in data['county_fips'].unique():
                data.loc[data['county_fips']==i,feature]=temp[i].tolist()
            return(data)
    def generate_country_covid_data(address):
        death = pd.read_csv(address+'international-covid-death-data.csv')
        confirmed = pd.read_csv(address+'international-covid-confirmed-data.csv')
        death=death[death['Country/Region']=='US'].T
        death=death.iloc[4:,:]
        death.iloc[1:]=(death.iloc[1:].values-death.iloc[:-1].values)
        death = death.reset_index()
        death.columns = ['date','death']
        death['death']= death['death'].astype(int)
        confirmed=confirmed[confirmed['Country/Region']=='US'].T
        confirmed=confirmed.iloc[4:,:]
        confirmed.iloc[1:]=(confirmed.iloc[1:].values-confirmed.iloc[:-1].values)
        confirmed = confirmed.reset_index()
        confirmed.columns = ['date','confirmed']
        confirmed['confirmed']= confirmed['confirmed'].astype(int)
        confirmed_death = pd.merge(death,confirmed)
        confirmed_death['county_fips']=1
        confirmed_death['date'] = confirmed_death['date'].apply(lambda x:datetime.datetime.strptime(x,'%m/%d/%y'))
        confirmed_death=confirmed_death.sort_values(by=['date'])
        confirmed_death['date'] = confirmed_death['date'].apply(lambda x:datetime.datetime.strftime(x,'%m/%d/%y'))
        return(confirmed_death)
    
    def get_csv(web_addres,file_address):
        url=web_addres
        print(url)
        req = requests.get(url)
        url_content = req.content
        csv_file = open(file_address, 'wb')
        csv_file.write(url_content)
        csv_file.close

    def get_updated_covid_data(address):
        get_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        ,address + 'international-covid-death-data.csv')
        get_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        ,address + 'international-covid-confirmed-data.csv')

    
    ##################################################################### imputation
#     get_updated_covid_data(address)
    timeDeapandantData = pd.read_csv(address + 'temporal-data.csv')

    if pivot == 'country':
        def country_aggregate(data):
            country_confirmed_death = generate_country_covid_data(address)
            data=country_confirmed_death
            return(data)

        timeDeapandantData = country_aggregate(timeDeapandantData)

    if only_death_flag:
        timeDeapandantData['county_fips'] = 1
        timeDeapandantData = timeDeapandantData[['date', 'county_fips', target]].copy()
    ##################################################################### cumulative mode

    if target_mode == 'cumulative':  # make target cumulative by adding the values of the previous day to each day
        timeDeapandantData = timeDeapandantData.sort_values(by=['date', 'county_fips'])

        dates = timeDeapandantData['date'].unique()
        for i in range(len(dates) - 1):
            timeDeapandantData.loc[timeDeapandantData['date'] == dates[i + 1], target] = \
                list(np.array(timeDeapandantData.loc[timeDeapandantData['date'] == dates[i + 1], target]) + \
                     np.array(timeDeapandantData.loc[timeDeapandantData['date'] == dates[i], target]))

    ###################################################################### weekly average mode

    if target_mode == 'weeklyaverage':  # make target weekly averaged

        def make_weekly(dailydata):
            dailydata['date'] = dailydata['date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
            # dailydata.drop(['weekend'],axis=1,inplace=True)
            # add weekday to find epidemic weeks
            dailydata['weekday'] = dailydata['date'].apply(lambda x: x.weekday())
            dailydata.sort_values(by=['date', 'county_fips'], inplace=True)
            dailydata = dailydata.reset_index(drop=True)
            numberofcounties = len(dailydata['county_fips'].unique())
            last_saturday_index = dailydata[dailydata['weekday'] == 5].last_valid_index()
            dailydata = dailydata.loc[:last_saturday_index, :]
            dailydata = dailydata.drop(['weekday'], axis=1)
            numberofweeks = len(dailydata['date'].unique()) // 7

            weeklydata = pd.DataFrame(columns=dailydata.columns.drop('date'))

            for i in range(numberofweeks):
                temp_df = dailydata.tail(numberofcounties * 7)  # weekly average of last week for all counties
                date = temp_df.tail(1)['date'].iloc[0]
                dailydata = dailydata.iloc[:-(numberofcounties * 7), :]
                sum_of_week = temp_df.groupby(['county_fips']).mean().reset_index()
                temp_df = temp_df.groupby(['county_fips']).mean().reset_index()
                temp_df[target] = sum_of_week[target]
                temp_df['date'] = numberofweeks - i  # week number
                weeklydata = weeklydata.append(temp_df)
            weeklydata.sort_values(by=['county_fips', 'date'], inplace=True)
            weeklydata = weeklydata.reset_index(drop=True)
            # weeklydata['date']=weeklydata['date'].apply(lambda x: datetime.datetime.strftime(x,'%m/%d/%y'))
            return (weeklydata)

        timeDeapandantData = make_weekly(timeDeapandantData)

    ###################################################################### weekly moving average mode
    if target_mode == 'weeklymovingaverage':
        def make_moving_weekly_average(dailydata):
            dailydata['date'] = dailydata['date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
            dailydata.sort_values(by=['date', 'county_fips'], inplace=True)
            numberofcounties = len(dailydata['county_fips'].unique())
            numberofdays = len(dailydata['date'].unique())
            dates = dailydata['date'].unique()

            weeklydata = pd.DataFrame(columns=dailydata.columns)

            while numberofdays >= 7:
                current_day_previous_week_data = dailydata.tail(numberofcounties * 7)
                current_day_data = dailydata.tail(numberofcounties)

                # weekly average of last week for all counties
                current_day_previous_week_data = current_day_previous_week_data.groupby(
                    ['county_fips']).mean().reset_index()
                # add weekly moving averaged target of lastweek to last day target
                current_day_data.loc[:, (target)] = current_day_previous_week_data.loc[:, (target)].tolist()

                weeklydata = weeklydata.append(current_day_data)
                dailydata = dailydata.iloc[:-(numberofcounties), :]  # remove last day for all counties from daily data
                numberofdays = numberofdays - 1
            weeklydata = weeklydata.sort_values(by=['county_fips', 'date'])
            weeklydata['date'] = weeklydata['date'].apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%y'))

            return (weeklydata)

        timeDeapandantData = make_moving_weekly_average(timeDeapandantData)

    ###################################################################### augmented weekly average mode
    if target_mode == 'augmentedweeklyaverage':
        print('augmentedweeklyaverage')

        def make_moving_weekly_average(dailydata):
            dailydata['date'] = dailydata['date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
            dailydata.drop(['weekend'], axis=1, inplace=True)
            # add weekday to find epidemic weeks
            dailydata['weekday'] = dailydata['date'].apply(lambda x: x.weekday())
            dailydata.sort_values(by=['date', 'county_fips'], inplace=True)
            dailydata = dailydata.reset_index(drop=True)
            last_saturday_index = dailydata[dailydata['weekday'] == 5].last_valid_index()
            first_monday_index = dailydata[dailydata['weekday'] == 6].first_valid_index()
            dailydata = dailydata.loc[first_monday_index:last_saturday_index, :]
            dailydata = dailydata.drop(['weekday'], axis=1)
            numberofdays = len(dailydata['date'].unique())
            numberofcounties = len(dailydata['county_fips'].unique())
            dates = dailydata['date'].unique()

            weeklydata = pd.DataFrame(columns=dailydata.columns)

            while numberofdays >= 7:
                current_day_previous_week_data = dailydata.tail(numberofcounties * 7)
                current_day_data = dailydata.tail(numberofcounties)
                date = current_day_data['date'].unique()[0]

                # weekly average of last week for all counties
                current_day_previous_week_data_features_mean = current_day_previous_week_data.groupby(
                    ['county_fips']).mean().reset_index()
                current_day_previous_week_data_features_mean.loc[:, ('date')] = date

                weeklydata = weeklydata.append(current_day_previous_week_data_features_mean)
                dailydata = dailydata.iloc[:-(numberofcounties), :]  # remove last day for all counties from daily data
                numberofdays = numberofdays - 1
            weeklydata = weeklydata.sort_values(by=['county_fips', 'date'])
            weeklydata['date'] = weeklydata['date'].apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%y'))

            return (weeklydata)

        timeDeapandantData = make_moving_weekly_average(timeDeapandantData)

    ###################################################################### differential target mode

    if target_mode == 'differential':  # make target differential
        reverse_dates = timeDeapandantData['date'].unique()[::-1]
        for index in range(len(reverse_dates)):
            date = reverse_dates[index]
            past_date = reverse_dates[index + 1]
            timeDeapandantData.loc[timeDeapandantData['date'] == date, target] = list(
                np.array(timeDeapandantData.loc[timeDeapandantData['date'] == date, target]) - np.array(
                    timeDeapandantData.loc[timeDeapandantData['date'] == past_date, target]))
            if index == len(reverse_dates) - 2:
                break
        timeDeapandantData.loc[timeDeapandantData[target] < 0, target] = 0

    ###################################################################### add future features

    if future_mode == True:
        def add_future_features(dailydata):
            # dailydata['date'] = dailydata['date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
            dailydata.sort_values(by=['date', 'county_fips'], inplace=True)
            dailydata = dailydata.reset_index(drop=True)
            numberofcounties = len(dailydata['county_fips'].unique())
            numberofdays = len(dailydata['date'].unique())
            new_data = pd.DataFrame(columns=dailydata.columns)
            futureDays = 2 if target_mode == 'weeklyaverage' else 14

            while numberofdays != 0:
                if numberofdays > futureDays:
                    # select the first day of all the counties
                    current_day_data = dailydata.head(numberofcounties).copy()
                    # select the next two weeks to compute their social-distancing average
                    next2weeks_data = dailydata.iloc[numberofcounties: numberofcounties * (futureDays + 1), :].copy()
                    # compute the average and round it
                    next2weeks_data = next2weeks_data.groupby(['county_fips']).mean()
                    # add the average of social-distancing of the next two weeks to the current day data
                    total_grade = 0
                    for temporal_feature in future_features:
                        if temporal_feature != 'social-distancing-total-grade':
                            future_feature = 'future-' + temporal_feature
                            current_day_data[future_feature] = next2weeks_data[temporal_feature].copy().tolist()
                            total_grade += current_day_data[future_feature]
                    current_day_data['future-social-distancing-total-grade'] = total_grade / 2
                else:
                    current_day_data = dailydata.head(numberofcounties).copy()
                    for temporal_feature in future_features:
                        future_feature = 'future-' + temporal_feature
                        current_day_data[future_feature] = np.NaN

                new_data = new_data.append(current_day_data.copy())
                # remove first day for all counties from daily data
                dailydata = dailydata.iloc[numberofcounties:].copy()
                numberofdays = numberofdays - 1

            new_data = new_data.sort_values(by=['county_fips', 'date'])
            # new_data['date'] = new_data['date'].apply(lambda x: datetime.datetime.strftime(x, '%m/%d/%y'))

            return (new_data)

        timeDeapandantData = add_future_features(timeDeapandantData)

    future_features = ["{}{}".format('future-', i) for i in future_features]

    ##################################################################

    if pivot != 'country':
        allData = pd.merge(independantOfTimeData, timeDeapandantData, on='county_fips')
    else:
        allData = timeDeapandantData

    allData = allData.sort_values(by=['date', 'county_fips'])
    allData = allData.reset_index(drop=True)

    step = 1
    if target_mode == 'augmentedweeklyaverage': step = 7

    nameOfTimeDependantCovariates = timeDeapandantData.columns.values.tolist()
    nameOfAllCovariates = allData.columns.values.tolist()

    result = pd.DataFrame()  # we store historical data in this dataframe
    totalNumberOfCounties = len(allData['county_fips'].unique())
    totalNumberOfDays = len(allData['date'].unique())

    # in this loop we make historical data
    for name in nameOfAllCovariates:
        # if covariate is time dependant
        if name in nameOfTimeDependantCovariates and name not in ['date', 'county_fips']:
            temporalDataFrame = allData[[name]]  # selecting column of the covariate that is being processed
            step = 1
            if target_mode == 'augmentedweeklyaverage': step = 7

            threshold = 0
            while threshold != h:
                # we dont want history for future features
                if name in future_features:
                    threshold = h - 1
                # get value of covariate that is being processed in first (totalNumberOfDays-h-r+1) days
                temp = temporalDataFrame.head(
                    (totalNumberOfDays + (step * (-h - r + 1))) * totalNumberOfCounties).copy().reset_index(drop=True)

                # we dont want date suffix for future features
                if name not in future_features:
                    temp.rename(columns={name: (name + ' t-' + str(h - threshold - 1))},
                                inplace=True)  # renaming column
                result = pd.concat([result, temp], axis=1)
                # deleting the values in first day in temporalDataFrame dataframe (similiar to shift)
                temporalDataFrame = temporalDataFrame.iloc[step * totalNumberOfCounties:]
                threshold += 1

        # if covariate is independant of time
        elif name not in nameOfTimeDependantCovariates and name not in ['date', 'county_fips']:
            # we dont need covariates that is fixed for each county in county mode
            # but also we need county and state name in all modes
            if (spatial_mode != 'county') or (name in ['county_name', 'state_name', 'state_fips']):
                temporalDataFrame = allData[[name]]
                temp = temporalDataFrame.head(
                    (totalNumberOfDays + (step * (-h - r + 1))) * totalNumberOfCounties).copy().reset_index(drop=True)
                result = pd.concat([result, temp], axis=1)

    # next 3 lines is for adding FIPS code to final dataframe
    temporalDataFrame = allData[['county_fips']]
    temp = temporalDataFrame.head(
        (totalNumberOfDays + (step * (-h - r + 1))) * totalNumberOfCounties).copy().reset_index(drop=True)
    result.insert(0, 'county_fips', temp)

    # next 3 lines is for adding date of day (t) to final dataframe
    temporalDataFrame = allData[['date']]
    temporalDataFrame = temporalDataFrame[totalNumberOfCounties * (step * (h - 1)):]
    temp = temporalDataFrame.head(
        (totalNumberOfDays + (step * (-h - r + 1))) * totalNumberOfCounties).copy().reset_index(drop=True)
    result.insert(1, 'date of day t', temp)

    # next 3 lines is for adding target to final dataframe
    temporalDataFrame = allData[[target]]
    temporalDataFrame = temporalDataFrame.tail(
        (totalNumberOfDays + (step * (-h - r + 1))) * totalNumberOfCounties).reset_index(drop=True)
    result.insert(1, 'Target', temporalDataFrame)
    for i in result.columns:
        if i.endswith('t-0'):
            result.rename(columns={i: i[:-2]}, inplace=True)

    result.dropna(inplace=True)

    ###################################################################### logarithmic target mode

    if target_mode == 'logarithmic':  # make target logarithmic
        result['Target'] = np.log((result['Target'] + 1).astype(float))

    ######################################################################

    result = result.sort_values(by=['county_fips', 'date of day t']).reset_index(drop=True)
    totalNumberOfDays = len(result['date of day t'].unique())
    county_end_index = 0
    overall_non_zero_index = list()
    for i in result['county_fips'].unique():
        county_data = result[result['county_fips'] == i]  # .reset_index(drop=True)
        county_end_index = county_end_index + len(result[result['county_fips'] == i])

        # we dont use counties with zero values for target variable in all history dates
        if (county_data[target + ' t'].sum() > 0):
            if h == 1:
                # find first row index with non_zero values for target variable in all history dates when history length<7
                first_non_zero_date_index = county_data[target + ' t'].ne(0).idxmax()
            elif h < 7:
                # find first row index with non_zero values for target variable in all history dates when history length<7
                first_non_zero_date_index = county_data[target + ' t-' + str(h - 1)].ne(0).idxmax()
            else:
                # find first row index with non_zero values for target variable in 7 last days of history when history length>7
                first_non_zero_date_index = county_data[target + ' t'].ne(0).idxmax() + 7

            zero_removed_county_index = [i for i in range(first_non_zero_date_index, county_end_index)]

            # in future mode we have more limited data, so we choose at least test_size days for test, validation and train sets
            if future_mode and len(zero_removed_county_index) >= 3 * test_size:
                overall_non_zero_index += zero_removed_county_index

            # we choose r days for test and r days for validation so at least we must have r days for train -> 3*r
            elif len(zero_removed_county_index) >= 3 * r:
                overall_non_zero_index = overall_non_zero_index + zero_removed_county_index

    zero_removed_data = result.loc[overall_non_zero_index, :]
    result = result.reset_index()
    # we use reindex to avoid pandas warnings
    zero_removed_data = result.loc[result['index'].isin(overall_non_zero_index), :]
    zero_removed_data = zero_removed_data.drop(['index'], axis=1)
    if pivot != 'country':
        result = zero_removed_data

    if pivot == 'state':
        temp = result
        temp['county_fips'] = temp['state_fips']
        temp['county_name'] = temp['state_name']
        temp.drop(['state_fips', 'state_name'], axis=1, inplace=True)
        all_columns = temp.columns.values
        social_distancing_columns = [col for col in all_columns if col.startswith('social-distancing')]
        confirmed_death_columns = [col for col in all_columns if
                                   col.startswith('confirmed') or col.startswith('death ')]
        base_columns = ['county_fips', 'county_name', 'date of day t']
        cumulative_columns = ['Target', 'total_population', 'meat_plants', 'passenger_load',
                              'area'] + confirmed_death_columns
        mean_columns = [col for col in all_columns if col not in cumulative_columns]
        cumulative_columns += base_columns

        cumulative_grouped = temp[cumulative_columns].groupby(base_columns)
        mean_grouped = temp[mean_columns].groupby(base_columns)

        cumulative_data = cumulative_grouped.sum()
        mean_data = mean_grouped.mean()

        temp = pd.DataFrame(data={}, columns=['county_fips', 'county_name', 'date of day t'])
        for name, group in cumulative_grouped:
            s = pd.Series(data=list(name), index=['county_fips', 'county_name', 'date of day t'])
            temp = temp.append(s, ignore_index=True)

        result = pd.concat([temp,
                            pd.DataFrame(mean_data.values, columns=mean_data.columns.values),
                            pd.DataFrame(cumulative_data.values, columns=cumulative_data.columns.values)],
                           axis=1)

        for social_distancing_column in social_distancing_columns:
            result[social_distancing_column] = result[social_distancing_column].map(lambda x: round(x))

    if 'index' in result.columns:
        result = result.drop(['index'], axis=1)
    if end_date != 0:
        result = result.iloc[:-(end_date), :]

    return result


def main():
    h = 0
    r = 1
    target = 'death'
    feature_selection = 'mrmr'
    spatial_mode = 'country'
    target_mode = 'weeklyaverage'
    address = './'
    pivot = 'country'
    future_features = []
    test_size = 21
    result = makeHistoricalData(0, r, test_size, target, 'mrmr', spatial_mode, target_mode, './',
                                   future_features, pivot, 0)
    # Storing the result in a csv file
    # result.to_csv('dataset_h=' + str(h) + '.csv', mode='w', index=False)


if __name__ == "__main__":
    main()