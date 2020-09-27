# base imports
import csv
import json
from datetime import date, timedelta

# self imports
import debug

# defines
_CSV_Directory_ = './csvFiles/'
_JSON_Directory_ = './jsonFiles/'

class handler_csv:
    def __init__(self):
        debug.debug_print("CSV Handler is up", 1)

    def _loadData(self, csvFilename):
        csvData = []
        csvFileFieldnames = []
        with open(_CSV_Directory_ + csvFilename) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            csvFileFieldnames = csvDriver.fieldnames
            for row in csvDriver:
                csvData.append(row)
        return (csvData, csvFileFieldnames)

    def _keyValues(self, dictionary, keys):
        returnList = []
        for key in keys:
            try:
                returnList.append(int(dictionary[key], 10))
            except:
                returnList.append(dictionary[key])
        return returnList

    def _isEqual(self, list1, list2):
        if len(list1) != len(list2):
            return 0
        for i in range(len(list1)):
            if list1[i] != list2[i]:
                return 0
        return 1

    # Function that merge two CSV files on 'commonColumn'
    def merge_csvFiles_addColumns(self, csvFilename1, csvFilename2, destinationFilename, file1Keys, file2Keys, newColumns):
        # Get csvFiles data
        csvData1, csvFile1Fieldnames = self._loadData(csvFilename1)
        csvData2 = self._loadData(csvFilename2)[0]
        
        # Generate new file fieldnames, add newClumns to fieldnames
        mergedDataFieldnames = csvFile1Fieldnames + newColumns

        # Merge and save data 
        with open(_CSV_Directory_ + destinationFilename, 'w') as destinationFile:
            csvDriver = csv.DictWriter(destinationFile, fieldnames=mergedDataFieldnames)
            csvDriver.writeheader()

            for row in csvData1:
                # Find correspondingRow on csvData2
                found = False
                for item in csvData2:
                    if self._isEqual(self._keyValues(row, file1Keys), self._keyValues(item, file2Keys)):
                        correspondingRow = {key:item[key] for key in newColumns}
                        found = True
                        break

                if found == False:
                    debug.debug_print("key not found:\t" + ', '.join(str(item) for item in self._keyValues(row, file1Keys)), 3)
                    continue

                # Generate and add row to file
                row.update(correspondingRow)
                csvDriver.writerow(row)
        
        debug.debug_print("SUCCESS: merge completed", 2)

    # Function that merge rows of two CSV files
    def merge_csvFiles_addRows(self, csvFilename1, csvFilename2, destinationFilename):
        # Get csvFiles data
        csvData1, csvFile1Fieldnames = self._loadData(csvFilename1)
        csvData2, csvFile2Fieldnames = self._loadData(csvFilename2)

        if csvFile1Fieldnames != csvFile2Fieldnames:
            debug.debug_print("ERROR: mismatch in columns", 2)
            return

        with open(_CSV_Directory_ + destinationFilename, 'w') as destinationFile:
            csvDriver = csv.DictWriter(destinationFile, fieldnames=csvFile1Fieldnames)
            csvDriver.writeheader()
            csvDriver.writerows(csvData1)
            csvDriver.writerows(csvData2)

        debug.debug_print("SUCCESS: merge completed", 2)

    # Load a CSV file, save interested fields known as 'fieldnames' to DestinationFilename, ignore others
    def simplify_csvFile(self, csvFilename, destinationFilename, fieldnames):
        # Get csvFile data
        csvData = self._loadData(csvFilename)[0]

        with open(_CSV_Directory_ + destinationFilename, 'w') as DestinationFile:
            csvDriver = csv.DictWriter(DestinationFile, fieldnames)
            csvDriver.writeheader()
            for row in csvData:
                csvDriver.writerow({k:row[k] for k in (fieldnames) if k in row})
        
        debug.debug_print("SUCCESS: simplify completed", 2)

class handler_json:
    data = []
    def __init__(self):
        debug.debug_print("JSON Handler is up", 1)

    def _loadData(self, jsonFilename):
        jsonMetaData = []
        with open(_JSON_Directory_ + jsonFilename) as jsonFile:
            jsonMetaData = json.load(jsonFile)
        return jsonMetaData

    # Function that transform a JSON file to CSV file. Design for hospitalBedMetaData
    def transform_jsonToCsv_hospitalBedData(self, jsonFilename, csvFilename):
        jsonMetaData = []
        fieldnames = ['county_fips', 'countyName', 'stateName', 'beds', 'unoccupiedBeds']
        jsonMetaData = self._loadData(jsonFilename)

        jsonCountiesData = jsonMetaData['objects']['counties']['geometries']
        singleCountyData = {}

        with open(_CSV_Directory_ + csvFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=fieldnames)
            csvDriver.writeheader()
            for row in jsonCountiesData:
                singleCountyData = {fieldnames[k]:(row['properties'][fieldnames[k]]).encode('utf-8') for k in range(1, len(fieldnames))}
                singleCountyData['county_fips'] = int(row['id'], 10)
                csvDriver.writerow(singleCountyData)
        
        debug.debug_print("SUCCESS: transform completed(hospitalBedData)", 2)

    # Function that transform a JSON file to CSV file. Design for socialDistancingBedMetaData
    def transform_jsonToCsv_socialDistancingData(self, jsonFilename, csvFilename):
        jsonMetaData = []
        countyFieldnames = ['stateFips', 'stateName', 'countyFips', 'countyName']
        dataFieldnames = ['date', 'totalGrade', 'visitationGrade', 'encountersGrade', 'travelDistanceGrade']
        jsonMetaData = self._loadData(jsonFilename)

        jsonCountiesData = jsonMetaData['hits']['hits']
        singleCountyData = {}

        with open(_CSV_Directory_ + csvFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=(countyFieldnames+dataFieldnames))
            csvDriver.writeheader()
            for county in jsonCountiesData:
                # set county data
                singleCountyData = {field:(county['_source'][field]).encode('utf-8') for field in countyFieldnames}
                # set Grade of county for each day, specified with 'date'
                for data in county['_source']['data']:
                    singleCountyData.update({field:data[field] for field in dataFieldnames})
                    csvDriver.writerow(singleCountyData)
        
        debug.debug_print("SUCCESS: transform completed(socialDistancingData)", 2)

    # Function that transform a JSON file to CSV file. Design for confirmAndDeathMetaData
    def transform_jsonToCsv_confirmAndDeathData(self, jsonFilename, csvFilename):
        startDay = date.fromisoformat('2020-01-22')
        jsonMetaData = []
        countyFieldnames = ['stateFIPS', 'stateAbbr', 'countyFIPS', 'county']
        dataFieldnames = ['date', 'confirmed', 'deaths']
        jsonMetaData = self._loadData(jsonFilename)

        singleCountyData = {}

        with open(_CSV_Directory_ + csvFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=(countyFieldnames+dataFieldnames))
            csvDriver.writeheader()

            numberOfDays = len(jsonMetaData[0]['deaths'])
            for day in range(numberOfDays):
                singleCountyData = {'date':(startDay+timedelta(days=day)).isoformat()}
                for county in jsonMetaData:
                    # pass invalid county
                    if int(county['countyFIPS'], 10) == 0:
                        continue
                    # set county data
                    singleCountyData.update({field:county[field] for field in countyFieldnames})
                    singleCountyData.update({field:county[field][day] for field in dataFieldnames if field in county})
                    # remove summation
                    if day != 0:
                        singleCountyData.update({field:(county[field][day] - county[field][day - 1]) for field in ['confirmed', 'deaths'] if field in county})
                    # write in file
                    csvDriver.writerow(singleCountyData)

        debug.debug_print("SUCCESS: transform completed(confirmAndDeathData)", 2)