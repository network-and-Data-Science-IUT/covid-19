# base imports
import requests
import json
import csv
# import progressbar

def debug_print(str, tier):
    print('\t'*tier + '|--' + str)

def debug_seprate():
    print('-'*30)


# self imports
import debug

# defines
_CSV_Directory_ = './csvFiles/'
_JSON_Directory_ = './jsonFiles/'

# An object for downloading data online and save to drive
class extractor:
    def __init__(self):
        debug_print("Extractor is up", 1)

    def get_socialDistancingData(self, stateFIPS, destinationFilename):
        jsonData = requests.get('https://covid19-scoreboard-api.unacastapis.com/api/search/covidcountyaggregates_v3?q=stateFips:%02d&size=4000' % (stateFIPS)).json()
        with open(_JSON_Directory_ + destinationFilename, 'w') as jsonFile:
            json.dump(jsonData, jsonFile)

        debug_print("SUCCESS: data extracted (socialDistancingData on stateFIPS:%02d)" % (stateFIPS), 2)

    def get_confirmAndDeathData(self, destinationFilename):
        jsonData = requests.get('https://usafactsstatic.blob.core.windows.net/public/2020/coronavirus-timeline/allData.json').json()
        with open(_JSON_Directory_ + destinationFilename, 'w') as jsonFile:
            json.dump(jsonData, jsonFile)

        debug_print("SUCCESS: data extracted (confirmAndDeathData)", 2)

    # A function that extract weather stations of all US counties
    def get_allStations(self, destinationFilename):
        countyData = []
        with open(_CSV_Directory_ + 'fixed-data.csv') as csvFile:
            csvDriver = csv.DictReader(csvFile)
            for row in csvDriver:
                countyData.append({'fips':row['county_fips'], 'name':row['county_name']})

        cookies = {
            'JSESSIONID': '54BF6C05EA3D53BF54F9349927E5CC62',
            '_ga': 'GA1.2.1721804174.1588253956',
            '_gid': 'GA1.2.850825076.1588253956',
            '_gat': '1',
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.ncdc.noaa.gov/cdo-web/datatools/selectlocation',
            'token': '0x2a',
            'X-Requested-With': 'XMLHttpRequest',
            'Connection': 'keep-alive',
        }

        fieldnames = ['county_fips', 'county_name', u'id', u'elevation', u'name', u'maxdate', u'datacoverage', u'longitude', u'latitude', u'elevationUnit', u'mindate']
        with open(_CSV_Directory_ + destinationFilename, 'a') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=fieldnames)

            numberOfCounties = len(countyData)
#             progressBarWidget = [progressbar.Percentage(),
#             ' ',
#             progressbar.Bar('#', '|', '|'),
#             ' ',
#             progressbar.Variable('County', width=12, precision=12),
#             ]
#             progressBar = progressbar.ProgressBar(maxval=numberOfCounties, widgets=progressBarWidget, redirect_stdout=True)
#             progressBar.start()

            step = 0
            try:
                logFile = open('station.log', 'r')
                step = int(logFile.read(), 10)
                logFile.close()
            except:
                logFile = open('station.log', 'w')
                logFile.write(str(step))
                logFile.close()
                csvDriver.writeheader()

            for i in range(step, numberOfCounties):
                with open('station.log', 'w') as logFile:
                    logFile.write(str(i))

                params = (
                    ('limit', '1000'),
                    ('datasetid', 'GHCND'),
                    ('locationid', 'FIPS:%05d' % (int(countyData[i]['fips'], 10))),
                    ('sortfield', 'name'),
                )

                try:
                    progressBar.update(i, County=countyData[i]['name'])
                    jsonData = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/stations', headers=headers, params=params, cookies=cookies).json()
                    stationData = {'county_name': countyData[i]['name'], 'county_fips': countyData[i]['fips']}
                    for row in jsonData['results']:
                        stationData.update(row)
                        csvDriver.writerow(stationData)

                except Exception as e:
                    print('[-] an error occurred while getting stations of {0} with fips: {1}'.format(countyData[i]['name'], countyData[i]['fips']))
                    print(e)

            progressBar.finish()

        debug_print("SUCCESS: data extracted (weather stations)", 2)

    def get_countyWeatherData(self, county_fips, stationID, startDate, endDate, destinationFilename):
        csvData = requests.get('https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&dataTypes=TAVG,TMAX,TMIN,PRCP,WT16,WT01,TSUN&stations={0}&startDate={1}&endDate={2}&boundingBox=90,-180,-90,180&format=csv'.format(stationID, startDate, endDate)).text
        csvData = (csvData.replace('"', '')).split('\n')

        with open(_CSV_Directory_ + destinationFilename, 'w') as csvFile:
            for row in csvData:
                # Frist step, add 'county_fips' to header
                if row == csvData[0]:
                    row = 'county_fips,' + row + '\n'
                # Other steps, add county_fips code to data
                elif row != '':
                    row = '{0},'.format(county_fips) + row + '\n'

                csvFile.write(row)

        debug_print("SUCCESS: data extracted (weatherData of station:{0})".format(stationID), 2)