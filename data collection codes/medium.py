# base imports
import csv
import os
# import progressbar

# self imports
import debug
from datetime import date
from handlers import handler_csv, handler_json
from extractor import extractor

# defines
_CSV_Directory_ = './csvFiles/'
_JSON_Directory_ = './jsonFiles/'

# Last step, don't need to complete it now
class mediumClass:
    jsonHandler = handler_json()
    csvHandler = handler_csv()
    downloadHandler = extractor()
    def __init__(self):
        
        debug.debug_print("Medium Class is up", 1)

    def generate_allSocialDistancingData(self):
        statesData = self.csvHandler._loadData('states.csv')[0]
        for state in statesData:
            fips = int(state['state_fips'], 10)
            self.downloadHandler.get_socialDistancingData(fips, 'temp.json')
            # First step, create socialDistancing.csv file
            if state == statesData[0]:
                self.jsonHandler.transform_jsonToCsv_socialDistancingData('temp.json', 'socialDistancing.csv')
            # Other steps, merge new data to socialDistancing.csv file
            else:
                self.jsonHandler.transform_jsonToCsv_socialDistancingData('temp.json', 'temp.csv')
                self.csvHandler.merge_csvFiles_addRows('socialDistancing.csv', 'temp.csv', 'socialDistancing.csv')

    # This functions remove useless stations from it's csv files. useless mean the stations that their max-date is less than 2020-1-22
    def clean_stations(self):
        stationsData = []
        fieldnames = []
        with open(_CSV_Directory_ + 'stations.csv') as csvFile:
            csvDriver = csv.DictReader(csvFile)
            fieldnames = csvDriver.fieldnames
            for row in csvDriver:
                stationsData.append(row)

        with open(_CSV_Directory_ + 'new_stations.csv', 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames)
            csvDriver.writeheader()
            startDay = date.fromisoformat('2020-01-22')
            for station in stationsData:
                try:
                    if date.fromisoformat(station['maxdate']) > startDay:
                        csvDriver.writerow(station)
                except:
                    continue

        debug.debug_print("SUCCESS: useless stations removed", 2)

    def generate_allWeatherData(self, startDate, endDate):
        stationsData = self.csvHandler._loadData('stations.csv')[0]

        numberOfStations = len(stationsData)
        # progressBarWidget = [progressbar.Percentage(),
        # ' ',
        # progressbar.Bar('#', '|', '|'),
        # ' ',
        # progressbar.Variable('FIPS', width=12, precision=12),
        # ' ',
        # progressbar.Variable('ID', width=12, precision=12),
        # ]
        # progressBar = progressbar.ProgressBar(maxval=numberOfStations, widgets=progressBarWidget, redirect_stdout=True)
        # progressBar.start()

        step = 0
        try:
            logFile = open('weather.log', 'r')
            step = int(logFile.read(), 10)
            logFile.close()
        except:
            logFile = open('weather.log', 'w')
            logFile.write(str(step))
            logFile.close()
        
        for i in range(step, numberOfStations):
            with open('weather.log', 'w') as logFile:
                logFile.write(str(i))

            stationID = stationsData[i]['id'].split(':')[1]
            countyFips = stationsData[i]['county_fips']
            # progressBar.update(i, FIPS=countyFips, ID=stationID)
            # First step, create weather.csv file
            if i == 0:
                self.downloadHandler.get_countyWeatherData(countyFips, stationID, startDate, endDate, 'new_weather.csv')
            # Other steps, merge new data to weather.csv file
            else:
                self.downloadHandler.get_countyWeatherData(countyFips, stationID, startDate, endDate, 'temp.csv')
                self.csvHandler.merge_csvFiles_addRows('new_weather.csv', 'temp.csv', 'new_weather.csv')

        # progressBar.finish()
        debug.debug_print("SUCCESS: data extracted (weather data)", 2)
