#!/usr/bin/python3
import pandas as pd
import numpy as np
import requests
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import datetime
# self imports
import debug
import handlers
import extractor
import medium

def get_csv(web_addres,file_address):
    url=web_addres
    print(url)
    req = requests.get(url)
    url_content = req.content
    csv_file = open(file_address, 'wb')
    csv_file.write(url_content)
    csv_file.close

if __name__ == "__main__":
    
    
    # get Social Distancing data
    
    mediumObject = medium.mediumClass()
    mediumObject.generate_allSocialDistancingData()
    
    
    weather=pd.read_csv('./csvFiles/weather.csv')
    weather=weather.dropna(subset=['DATE'])
    weather['DATE']=weather['DATE'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
    startdate = datetime.datetime.strftime(max(weather['DATE'] - datetime.timedelta(days=10)) ,'%Y-%m-%d')
    today = datetime.datetime.now()
    enddate = datetime.datetime.strftime(today ,'%Y-%m-%d')
    
    # get weather data

    mediumObject = medium.mediumClass()
    mediumObject.downloadHandler.get_countyWeatherData('1001', 'USW00093228', startdate, enddate, 'test.csv')
    mediumObject.generate_allWeatherData(startdate, enddate)

    
    # get confirmed cases data
    get_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv',\
        './csvFiles/covid_confirmed_cases.csv')
    # get deaths data
    get_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv',\
            './csvFiles/covid_deaths.csv')
    # get tests data
    get_csv('https://covidtracking.com/api/v1/states/daily.csv',\
            './csvFiles/daily-state-test.csv')
    
    ########################### add new weather to weather file
    new_weather=pd.read_csv('csvFiles/new_weather.csv')
    weather=pd.read_csv('csvFiles/weather.csv')
    weather=weather.append(new_weather)
    weather=weather.drop_duplicates(subset=['county_fips','STATION','DATE'])
    weather.to_csv('csvFiles/weather.csv', index=False)
    
    ########################################################################## concat and prepare data
    
    fix=pd.read_csv('csvFiles/fixed-data.csv')
    socialDistancing=pd.read_csv('csvFiles/socialDistancing.csv')
    cof=pd.read_csv('csvFiles/covid_confirmed_cases.csv')
    
    

    # max date recorded
    confirmed_and_death_max_date = max([datetime.datetime.strptime(x,'%m/%d/%y') for x in cof.columns[4:]]).date()

    # preprocess socialDistancing
    socialDistancing=socialDistancing[['countyFips', 'date',
           'totalGrade', 'visitationGrade', 'encountersGrade',
           'travelDistanceGrade']]
    socialDistancing=socialDistancing.rename(columns={'countyFips':'county_fips'})
    socialDistancing['county_fips']=socialDistancing['county_fips'].apply(lambda x:x[2:7]).astype(int)
    socialDistancing['date']=socialDistancing['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    # max date recorded
    socialDistancing_max_date = max(socialDistancing['date']).date()

    # find dates which confirmed cases and deaths are recorded
    valid_dates=cof.columns[4:].tolist()

    ################################################################### create template for data

    fips=pd.DataFrame(columns=['fips'])
    fips['fips']=fix['county_fips'].tolist()*len(valid_dates)
    fips.sort_values(by='fips',inplace=True)
    data=pd.DataFrame(columns=['county_fips','date'])
    data['county_fips']=fips['fips']
    data['date']=valid_dates*3142 
    data['date']=data['date'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%y'))
    data.sort_values(by=['county_fips','date'],inplace=True)

    ################################################################### add socialDistancing data

    data=pd.merge(data,socialDistancing,how='left',left_on=['county_fips','date'],right_on=['county_fips','date'])
    data=data.rename(columns={'totalGrade':'social-distancing-total-grade','visitationGrade':'social-distancing-visitation-grade',
                'encountersGrade':'social-distancing-encounters-grade','travelDistanceGrade':'social-distancing-travel-distance-grade'})


    ################################################################ add test

    dailytest = pd.read_csv('./csvFiles/daily-state-test.csv')
    dailytest['date']=dailytest['date'].astype(str).apply(lambda x:datetime.datetime.strptime(x,'%Y%m%d'))
    dailytest=dailytest[['date','fips','totalTestResultsIncrease']]

    test_max_date = max(dailytest['date']).date()

    data['fips']=data['county_fips']//1000
    data=pd.merge(data,dailytest,how='left',left_on=['date','fips'],right_on=['date','fips'])

    data.drop(['fips'],axis=1,inplace=True)
    data.rename(columns={'totalTestResultsIncrease':'daily-state-test'},inplace=True)

    ############################################################## add weather data

    weather=pd.read_csv('csvFiles/weather.csv',na_values=np.nan)

    weather=weather.dropna(subset=['DATE'])
    weather['DATE']=weather['DATE'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    weather_max_date = max(weather['DATE'].unique())
    if isinstance(weather_max_date,np.datetime64):
        weather_max_date = datetime.datetime.utcfromtimestamp(weather_max_date.tolist()/1e9).date()
    elif isinstance(weather_max_date,pd._libs.tslibs.timestamps.Timestamp):
        weather_max_date = weather_max_date.date

    perc=weather.dropna(subset=['PRCP'])[['county_fips','DATE','PRCP']]
    perc.drop_duplicates(subset=['county_fips','DATE'],inplace=True)
    data=pd.merge(data,perc,how='left',left_on=['county_fips','date'],right_on=['county_fips','DATE'])
    data.drop(['DATE'],axis=1,inplace=True)
    data.rename(columns={'PRCP':'precipitation'},inplace=True)

    # impute average using min and max values
    rows_with_null_average=weather[(pd.isnull(weather['TAVG']))&(~pd.isnull(weather['TMAX']))&(~pd.isnull(weather['TMIN']))].index.tolist()
    weather.loc[rows_with_null_average,'TAVG']=weather.loc[rows_with_null_average,'TMAX']+weather.loc[rows_with_null_average,'TMIN']
    weather.loc[rows_with_null_average,'TAVG']=weather.loc[rows_with_null_average,'TAVG']/2
    weather.loc[rows_with_null_average,'TAVG']=weather.loc[rows_with_null_average,'TAVG'].round()

    temperature=weather.copy()[['county_fips', 'DATE','TAVG']]
    temperature.dropna(subset=['TAVG'],inplace=True)
    temperature.drop_duplicates(subset=['county_fips','DATE'],inplace=True)
    data=pd.merge(data,temperature,how='left',left_on=['county_fips','date'],right_on=['county_fips','DATE'])
    data.drop(['DATE'],axis=1,inplace=True)
    data.rename(columns={'TAVG':'temperature'},inplace=True)
    # recorrect scale
    data['temperature']=data['temperature']/10

    ############################################################# add confirmed cases and deaths

    cof=pd.read_csv('./csvFiles/covid_confirmed_cases.csv')
    det=pd.read_csv('./csvFiles/covid_deaths.csv')

    # derive new cases from cumulative cases
    cof2=cof.copy()
    for i in range(5,cof.shape[1]):
        cof2.iloc[:,i]=cof.iloc[:,i]-cof.iloc[:,i-1]
    cof=cof2.copy()

    det2=det.copy()
    for i in range(5,det.shape[1]):
        det2.iloc[:,i]=det.iloc[:,i]-det.iloc[:,i-1]
    det=det2.copy()

    cof=cof[cof['countyFIPS'].isin(data['county_fips'])]
    det=det[det['countyFIPS'].isin(data['county_fips'])]

    # add new cases to data
    data=data.drop_duplicates(subset=['county_fips','date'])

    for i in cof.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        data.loc[data['date']==j,'confirmed']=cof[i].copy().tolist()

    for i in det.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        data.loc[data['date']==j,'death']=det[i].copy().tolist()

    # save unimputed data
    unimputed_data=data.copy()

    # impute negative number of confirmed cases and deaths

    reverse_dates=cof.columns[4:][::-1]
    while cof.iloc[:,4:].min().sum()<0:
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            negative_index=cof[cof[date]<0].index
            cof2.loc[negative_index,date] = 0
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            past_date = reverse_dates[i+1]
            negative_index=cof[cof[date]<0].index
            cof2.loc[negative_index,past_date] = cof2.loc[negative_index,past_date]+cof.loc[negative_index,date]
            if i==len(reverse_dates)-2:
                break
        cof=cof2.copy()

    reverse_dates=det.columns[4:][::-1]
    while det.iloc[:,4:].min().sum()<0:
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            negative_index=det[det[date]<0].index
            det2.loc[negative_index,date] = 0
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            past_date = reverse_dates[i+1]
            negative_index=det[det[date]<0].index
            det2.loc[negative_index,past_date] = det2.loc[negative_index,past_date]+det.loc[negative_index,date]
            if i==len(reverse_dates)-2:
                break
        det=det2.copy()

    cof=cof[cof['countyFIPS'].isin(data['county_fips'])]
    det=det[det['countyFIPS'].isin(data['county_fips'])]

    # add imputed values to data

    for i in cof.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        data.loc[data['date']==j,'confirmed']=cof[i].copy().tolist()

    for i in det.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        data.loc[data['date']==j,'death']=det[i].copy().tolist()

    ########################################################################## add virus pressure to data

    adjacency_matrix=pd.read_csv('./csvFiles/adj_mat.csv')
    adjacency_matrix.index=adjacency_matrix['Unnamed: 0']
    adjacency_matrix.drop('Unnamed: 0',axis=1,inplace=True)

    data['virus-pressure']=0

    confirmed=pd.DataFrame(index=data['county_fips'].unique(),columns=data['date'].unique())

    for i in confirmed.columns:
        confirmed[i]=data.loc[data['date']==i,'confirmed'].tolist()
    for i in confirmed.columns:
        confirmed[i]=confirmed[i]
    for i in data['date'].unique():
        data.loc[data['date']==i,'virus-pressure']=np.dot(adjacency_matrix,confirmed[[i]])

    adjacency_matrix['neighbur_count'] = adjacency_matrix.sum().values.tolist()

    adjacency_matrix['county_fips']=adjacency_matrix.index
    adjacency_matrix.loc[adjacency_matrix['neighbur_count']==0,'neighbur_count']=1
    data=pd.merge(data,adjacency_matrix[['neighbur_count','county_fips']])
    data['virus-pressure']=data['virus-pressure']/data['neighbur_count']

    data=data.drop(['neighbur_count'],axis=1)


    ###################################################################### 
    # find max date with all features recorded and save unimputed data

    max_date = min(weather_max_date,socialDistancing_max_date,confirmed_and_death_max_date,test_max_date)
    max_date = datetime.datetime.combine(max_date, datetime.datetime.min.time())

    data=data[data['date']<=max_date]
    unimputed_data=unimputed_data[unimputed_data['date']<=max_date]

    # data['date']=data['date'].apply(lambda x: x.strftime('%m/%d/%y'))
    unimputed_data['date']=unimputed_data['date'].apply(lambda x: x.strftime('%m/%d/%y'))
    unimputed_data.to_csv('csvFiles/unimputed-temporal-data.csv',index=False)
    
    raw_data=pd.merge(unimputed_data,fix,how='left')
    raw_data.to_csv('raw-data.csv',index=False)

    ########################################################################## imputation

    # data['date']=data['date'].apply(lambda x:datetime.datetime.strptime(x,'%m/%d/%y'))

    covariate_to_imputed = ['social-distancing-total-grade','social-distancing-visitation-grade',
                            'social-distancing-encounters-grade','social-distancing-travel-distance-grade',
                            'precipitation','temperature']

    # save counties with all nulls
    temp=data.groupby('county_fips').count()
    counties_with_all_null_ind={}
    for i in covariate_to_imputed:
        counties_with_all_null_ind[i]=temp[temp[i]==0].index.tolist()
        
    
    
    # prepare social distancing features for imputation
    
    data['social-distancing-total-grade']=data['social-distancing-total-grade'].replace(['A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F']\
                                                                                        ,[12,11,10,9,8,7,6,5,4,3,2,1])

    for i in ['social-distancing-visitation-grade','social-distancing-encounters-grade',\
              'social-distancing-travel-distance-grade']:
        data[i]=data[i].replace(['A','B','C','D','F'],[5,4,3,2,1])


    # impute first days social distancing with lowest value    
    social_distancing_grades = ['social-distancing-total-grade','social-distancing-visitation-grade',
                                'social-distancing-encounters-grade','social-distancing-travel-distance-grade']

    for social_distancing_grade in social_distancing_grades:
        data.loc[(data['date']<datetime.datetime(2020,2,24)),social_distancing_grade]=1 # we have no social distancing data before 2020,2,24

    data['date']=data['date'].apply(lambda x: x.strftime('%m/%d/%y'))
    


    # impute covariates with KNN imputer

    for covar in covariate_to_imputed:

        print(covar,' null count:',len(counties_with_all_null_ind[covar]))

        temp=pd.DataFrame(index=data['county_fips'].unique().tolist(),columns=data['date'].unique().tolist())

        for i in data['date'].unique():
            temp[i]=data.loc[data['date']==i,covar].tolist()

        X = np.array(temp)
        imputer = KNNImputer(n_neighbors=5)
        imp=imputer.fit_transform(X)
        imp=pd.DataFrame(imp)
        imp.columns=temp.columns
        imp.index=temp.index
        for i in data['date'].unique():
            data.loc[data['date']==i,covar]=imp[i].tolist()

        # delete values for all null counties
        data.loc[data['county_fips'].isin(counties_with_all_null_ind[covar]),covar]=np.NaN
        
    # transform social distancing features to nominal grades
    data['social-distancing-total-grade']=data['social-distancing-total-grade'].replace([12,11,10,9,8,7,6,5,4,3,2,1]\
                                                                                        ,['A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F'])

    for i in ['social-distancing-visitation-grade','social-distancing-encounters-grade',\
              'social-distancing-travel-distance-grade']:
        data[i]=data[i].replace([5,4,3,2,1],['A','B','C','D','F'])

    # remove social-distancing-visitation-grade from imputed data cause it has high volume of nulls
    data=data.drop(['social-distancing-visitation-grade'],axis=1)

    # impute state daily test

    first_day_null = data.loc[pd.isnull(data['daily-state-test'])].index

    data.loc[data['daily-state-test']<0,'daily-state-test']=np.NaN
    value_count=data.groupby('county_fips').count()
    counties_with_all_nulls=value_count[value_count['daily-state-test']==0]
    temp=pd.DataFrame(index=data['county_fips'].unique().tolist(),columns=data['date'].unique().tolist())

    for i in data['date'].unique():
        temp[i]=data.loc[data['date']==i,'daily-state-test'].tolist()
    X = np.array(temp)
    imputer = KNNImputer(n_neighbors=5)
    imp=imputer.fit_transform(X)
    imp=pd.DataFrame(imp)
    imp.columns=temp.columns
    imp.index=temp.index
    for i in data['date'].unique():
        data.loc[data['date']==i,'daily-state-test']=imp[i].tolist()
    if(len(counties_with_all_nulls)>0):
        data.loc[data['county_fips'].isin(counties_with_all_nulls.index),'daily-state-test']=np.NaN

    data=data.sort_values(by=['county_fips','date'])
    data.to_csv('csvFiles/temporal-data.csv',index=False)

    #################################################### remove counties with all nulls for some features

    fixed_features_with_nulls=['ventilator_capacity','icu_beds','deaths_per_100000']

    for i in fixed_features_with_nulls:
        nullind=fix.loc[pd.isnull(fix[i]),'county_fips'].unique()
        
        data=data[~data['county_fips'].isin(nullind)]
        fix=fix[~fix['county_fips'].isin(nullind)]

    timeDeapandant_features_with_nulls=['social-distancing-travel-distance-grade','social-distancing-total-grade',
                                        'social-distancing-encounters-grade','temperature','precipitation']

    for i in timeDeapandant_features_with_nulls:
        nullind=data.loc[pd.isnull(data[i]),'county_fips'].unique()
        
        data=data[~data['county_fips'].isin(nullind)]
        fix=fix[~fix['county_fips'].isin(nullind)]


    ##### saivng final imputed data 
    fix.to_csv('csvFiles/full-fixed-data.csv', index=False)
    data.to_csv('csvFiles/full-temporal-data.csv', index=False)
    
    all_data=pd.merge(data,fix,how='left')
    all_data.to_csv('imputed-data.csv',index=False)
    
    


    
