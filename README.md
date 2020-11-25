# Dataset of COVID-19 outbreak and potential predictive features in the USA

	Data fields description:
	
		county_fips:
				County FIPS code
        
    	state_fips:
				State FIPS code
        
		county_name:
				Name of county
        
		state_name:
				Name of state
        
		total_population:
				Total population of each county
				
		percent_population_in_state:
				Percentage of the county population in the state
        
		female-percent:
				percent of female residents in each county
				source: https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/cc-est2018-alldata.csv
		age distribution:
				percent of residents in different age groups
				Consisted of:
					'age_0_4', 'age_5_9', 'age_10_14', 'age_15_19',
      					'age_20_24', 'age_25_29', 'age_30_34', 'age_35_39', 'age_40_44',
       					'age_45_49', 'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69',
       					'age_70_74', 'age_75_79', 'age_80_84', 'age_85_or_higher'
				Sorrce: https://www.census.gov/programs-surveys/popest.html
		area:
				The land area in Miles^2
				Source: https://www2.census.gov/library/publications/2011/compendia/usa-counties/excel/LND01.xls (Data from census)
				
		population_density:
				Calculated as "population" / "area"
		latitude:
				Latitude of the county center
		longitude:
				Longitude of the county center
		hospital_beds_ratio:
				number of hospital beds per person
				Calculated as "beds(per1000)" / 1000
				"beds(per1000)" is extracted from hospital-beds.csv
				source: https://www.urban.org/policy-centers/health-policy-center/projects/understanding-hospital-bed-capacities-nationwide-amid-covid-19
		icu_beds_ratio:
				number of icu beds per person
				sourse: https://public.tableau.com/profile/todd.bellemare#!/vizhome/DefinitiveHCCOVID-19CapacityPredictor/DefinitiveHealthcareCOVID-19CapacityPredictor		
		ventilator_capacity_ratio:
				number of ventilator per person
				sourse: https://public.tableau.com/profile/todd.bellemare#!/vizhome/DefinitiveHCCOVID-19CapacityPredictor/DefinitiveHealthcareCOVID-19CapacityPredictor

		median_household_income:
				Median household income in the Past 12 Months
				source: https://www.census.gov/programs-surveys/saipe.html
		gdp_per_capita:
				gross domestic product per capita for each county (economic measure)
				source: https://ssti.org/blog/useful-stats-10-year-changes-real-gdp-county-and-industry-2009-2018
					https://www.bea.gov/data/gdp/gdp-county-metro-and-other-areas

		houses_density:
				Calculated as "number of houses" / "land area"
				source: https://www2.census.gov/programs-surveys/popest/tables/2010-2018/housing/totals/PEP_2018_PEPANNHU_counties.zip
		education_level:
				represent the percentage of residents with different levels of education
				consisted of :
					'less_than_high_school_diploma'
					'high_school_diploma_only'
       					'some_college_or_higher'
				source: https://www.ers.usda.gov/webdocs/DataFiles/48747/Education.xls?v=568.3
				
		total_college_population:
				Total number of students and staff of universities and colleges devided by total population in each county
				source: https://nces.ed.gov/ipeds/use-the-data
				
		immigrant_student_ratio:
				Total number of students who study in this county but are residents of the other states, divided by the total county population
				Source: https://nces.ed.gov/ipeds/use-the-data
			
		percent_diabetes:
				Percent of Adults with Diabetes in each county
				Source: https://www.countyhealthrankings.org/app/alabama/2020/downloads
		percent_smokers:
				Percent of smokers in each county
				Source: https://www.countyhealthrankings.org/app/alabama/2020/downloads
		percent_insured:
				Percentage of health insured residents
				Source: https://www.countyhealthrankings.org/app/alabama/2020/downloads
				
		Religious_congregation_ratio:
				Percent of religion congregation members in each county
				Source: http://www.thearda.com/Archive/Files/Downloads/RCMSCY10_DL.asp
		political_party:
				Political party for each state (set to 0 for republican & 1 for democratic)
				Source: https://en.wikipedia.org/wiki/List_of_United_States_governors

		airport_distance:
				distance to nearest international airport with average daily passenger load more than 10
				Source: https://catalog.data.gov/dataset/airports & https://openflights.org/data.html
				
		passenger_load_ratio:
				average daily passenger load of nearest international airport to the county
				Source: https://data.transportation.gov/Aviation/International_Report_Passengers/xgub-n9bw
				
		deaths_per_100000:
				county 2018 deaths per 100000
				Source: https://wonder.cdc.gov/mcd-icd10.html
				
	
		meat_plants:
				Number of meat and poultry processing plants in each county
				Source: https://origin-www.fsis.usda.gov/wps/wcm/connect/3e414e13-d601-427c-904a-319e2e84e487/MPI_Directory_by_Establishment_Name.xls?MOD=AJPERES

    		weather_data:
				consisted of:
					'Precipitation'
					'Temperature'
				
				source: https://www.ncdc.noaa.gov/cdo-web/datatools/selectlocation
				We find stations once from source page, then we used them and API to gather weather data
				API page: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
				
		social-distancing-grades:
				measure of social distancing adherence in each county
				consisted of:
					'social-distancing-visitation-grade'
					'social-distancing-encounters-grade'
					'social-distancing-travel-distance-grade'
								Values are (A,B,C,D,F)
									
					 and 'social-distancing-total-grade'
								Values are (A,A-,B+,B,B-,C+,C,C-,D+,D,D-,F)
								
				Source: https://unacast-2019.webflow.io/covid19/social-distancing-scoreboard
		
		mobility-trend-change:
		
				Percent change in mobility trends in different place categories compared to pre-COVID-19 period
				consisted of:
					'retail_and_recreation_mobility_percent_change'
					'grocery_and_pharmacy_mobility_percent_change'
					'parks_mobility_percent_change'
					'transit_stations_mobility_percent_change'
					'workplaces_mobility_percent_change,residential_mobility_percent_change'
					
				Source: https://www.google.com/covid19/mobility/
				
		virus-pressure:
				Calculated as Average("covid_19_confirmed_cases") over neighboring counties

		daily-state-test:
				Number of daily tests performed in each state
				Source:https://covidtracking.com
		covid_19_confirmed_cases:
				number of confirmed covid-19 cases
				Source: https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/
		covid_19_deaths:
				number of covid-19 deaths
				Source: https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/
