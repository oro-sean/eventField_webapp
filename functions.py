def find_starts(rawLog):
	## Move to more generic log manipulations file
	## make threshold and time after start to query default keyword args
	
	"""
	find_starts takes a raw log file and returns a dictionary of race starts.
	The dictionary keys are the start numbers and each key has a list with ther following atributes
	- Time stamp (at gun)
	- Pt Lat (at gun)
	- Pt Lon (at gun)
	- Stb Lat (at gun)
	- Stb Lon (at gun)
	- Mk Lat (5 seconds after)
	- Mk Lon (5 seconds after)
	
	The start is also labeled a recall or race depending on hoe soon another start is found.
	
	The function looks for columns named as follows
	
	Tm2Gn - race timer (in seconds)
	TimeStamp - time of day (pandas datetime object)
	PtLat
	PtLon
	StbLat
	StbLon
	MkLat
	MkLon
	
	"""
	
    import statistics as stats
    import numpy as np

    threshold_newRace = 1800 # set threshold for new race in seconds
    ## search for start times and generate a dictionary for each start
    cases = [] # create an empty list to store potential start times in
    startCount = [] # create list to store the number of starts in the log file
    Tm2Gn = rawLog['Tm2Gn'].dropna().astype('int') # create pandas series of the expedition count down variable
    cases.append(list(Tm2Gn[Tm2Gn == 0].index)) # append the index of the times that the countdown variable is 0
    for i in range(15): # iterate over 1 to 15 seconds before the stat and record the indices that corresponds to 0 for each case
        cases.append([x + i for x in list(Tm2Gn[Tm2Gn == i].index)])
    for n in range(len(cases)): # iterate over each case
        case = cases[n] # create the individual case
        for i in range(len(case)-1): # iterate over each start form each case
            if i > len(case)-2: # as entries are deleated check the number of remaining entries is less than the index value
                break # if not enough entries leave case
            else:
                if abs(case[i]-case[i+1]) < 5: # if multiple possible start times exist remove them
                    del case[i+1]
        startCount.append(len(case)) # append the number of valid starts found to the start count

    startCount = stats.mode(startCount) # take the most common number of starts found as the correct number of starts
    n = 0 # set counter to 0
    while n < len(cases): # iterate over each case using while loop as the number will change as cases are eliminated
        if len(cases[n]) != startCount: # deleate cases that do not contain the correct number of starts
            del cases[n]
            n -= 1 # if a case is deleated reduce counter by 1 so as the subsequent case is examined
        n += 1 # increase counter by 1
    startDict = {} # create an empty dictionary to enter start attributes into
    # dict lists index, start time, port lat, port long, stb lat, stb long, active mark 5 seconds after the start
    cases = np.array(cases).T.tolist() # transpose the cases list for easy access when iterating over
    for i in range(startCount): # iterate over each start time
        startDict[i+1] = stats.mode(cases[i]) # take start index as the most common start index, add one so as first start is 1 not 0
    varLabels = ['TimeStamp', 'PtLat', 'PtLon', 'StbLat', 'StbLon'] # list of labels for required variables
    for key in startDict.keys(): # iterate over each start in the start dictionary
        attributes = [] # create empty list to store attributes in
        attributes.append(startDict[key]) # store the index so as it isn't lost
        for var in varLabels: # iterate over variables of interest
            attributes.append(rawLog[var][startDict[key]]) # store other attributes
        for var in ['MkLat', 'MkLon']: # add active mark lat and long 5 seconds after start
            attributes.append(rawLog[var][startDict[key]+5])

        startDict[key] = attributes # assign list of attributes to the start dictionary

    i = 0 # initiate counter to prevent checking last start
    for key in startDict.keys(): # iterate of each key
        i += 1 # increase counter by 1
        if i < len(startDict.keys()): # check not last entry
            if (startDict[key+1][1] - startDict[key][1]).total_seconds() < threshold_newRace: # check time of following start
                startType = 'Recal' # if within 15min designate recall
            else:
                startType = 'Race' # otherwise type is race
        else:
            startType = 'Race' # last start is always assumed race

        startDict[key].append(startType) #append the start type to the start dictionary

    return startDict
    ###
    #The start dictionary contains
    #key - start number
    # value [df index, Start time, port Lat, port long, stb lat, stb long, mark 1 lat, mark 1 long, start type]

def PrepLog(logPath):
	## move to more generic logmanipulation file
	
	"""
	The PrepLog function takes a file path and loads a csv log file.
	The old expedition log format was the template for this function but any csv with a header row and a row per observation will work.
	
	The column headers are then relabelled considering common synomons (ie "UTC" becomes "TimeStamp")
	
	The timestamp column is changed to a pandas datetime object assuming an original format of expedtion time.
	the race counter is changed to float number in seconds (assuming it was in decimals of days - expedition standard)
	
	a pandas data fram with all raw log data as float64 type is returned (except timestamp - pandas date time)
	
	
	"""
    import pandas as pd

    ## Import log and convert to generic veering format

    ## define standard variable names and synomons
    varSyn = {'TimeStamp': ['UTC', 'BST', 'AEST', 'AEDT'],
          'Bsp': ['BSP'],
          'Hdg': ['HDG'],
          'Heel': ['HEEL'],
          'Trim': ['TRIM'],
          'Cog': ['COG'],
          'Sog': ['SOG'],
          'Lat': ['LAT', 'LATTITUDE'],
          'Lon': ['LON', 'LONGITUDE'],
          'Awa': ['AWA'],
          'Aws': ['AWS'],
          'Twa': ['TWA'],
          'Tws': ['TWS'],
          'Twd': ['TWD'],
          'MkLat': ['MK LAT'],
          'MkLon': ['MK LON'],
          'PtLat': ['PORT LAT'],
          'PtLon': ['PORT LON'],
          'StbLat': ['STBD LAT'],
          'StbLon': ['STBD LON'],
          'Tm2Gn': ['TMTOGUN']
          }

    rawLog = pd.read_csv(logPath, low_memory=False) # import logfile to pandas DF

    renameDict = {}
    for varName in varSyn.keys():
        varFound = False
        for syn in varSyn[varName]:
            for logVar in rawLog.columns:
                if syn == logVar.upper():
                    renameDict[logVar] = varName
                    varFound = True
                    break
        if not varFound:
            print("No variable was found for "+varName)

    rawLog = rawLog.rename(columns=renameDict)

    for var in list(rawLog): # iterate over each variable and change to numeric and float type
        rawLog[var] = pd.to_numeric(rawLog[var], errors='coerce') # make each variable a numeric variables
        rawLog[var] = rawLog[var].astype('float') # make data type float

    rawLog['TimeStamp'] = pd.to_datetime(rawLog['TimeStamp'], unit='D', origin='1899-12-30').round('1s') # change time variable to pandas datetime type
    rawLog['Tm2Gn'] = rawLog['Tm2Gn']*86400 # create Tm2Gn variable converting decimals of days to seconds
    rawLog.fillna(method='ffill', inplace=True)

    return rawLog

def FindFeature(raceFrame, startIndex, startNo):
	## good potential for a class
	## tidy up logging so as it can be used
	"""
	Find race features takes a pandas data frame consisitng of all observation from one start until the next start.
	The index and start no are also handed over.
	the raceFrame also has a column 'Dist' which is the eucledian distance of the lat and long between the stating reference point
	and the boats current position
	
	Find features looks for peaks and troughs in this distance and the gradient of the distance cure before and after.
	It then tries to build a logical race orderted Top, botttom top ect ect to finish
	
	a dictionary of race features is returned as well as a log
	
	"""
    import numpy as np
    import scipy.signal as signal

    featureDict = {}
    for type in ['peaks', 'troughs']: # repeat process looking for peaks and troughs
        if type == "peaks":
            features = signal.find_peaks(raceFrame['Dist'], distance= 300, width=60)
        elif type == "troughs":
            features = signal.find_peaks(-raceFrame['Dist'], distance= 300, width=60)
        else:
            print("Error - Type not identified")

        grad = np.gradient(raceFrame['Dist'].rolling(10).mean())

        for featNo in range(len(features[0])): # iterate over each feature
            if type == "peaks":
                before = grad[features[0][featNo]-300:features[0][featNo]] > 0 #check if gradient in 5min before is greater than zero ie going away from start
                after = grad[features[0][featNo]:features[0][featNo]+300] < 0 # check if gradient in 5 min after is less than zero ie going away from mark
                if after.sum() > 269 and before.sum() > 269:
                    featureDict[startIndex+ features[0][featNo]] = ['Top', features[0][featNo], startIndex+ features[0][featNo] ]
                else:
                    featureDict[startIndex+ features[0][featNo]] = ['Finish', features[0][featNo], startIndex+ features[0][featNo] ]
            elif type == "troughs":
                before = grad[features[0][featNo]-300:features[0][featNo]] < 0 #check if gradient in 5min before is greater than zero ie going away from start
                after = grad[features[0][featNo]:features[0][featNo]+300] > 0 # check if gradient in 5 min after is less than zero ie going away from mark
                if after.sum() > 269 and before.sum() > 269:
                    featureDict[startIndex+ features[0][featNo]] = ['Bottom', features[0][featNo], startIndex+ features[0][featNo] ]
                else:
                    featureDict[startIndex+ features[0][featNo]] = ['Finish', features[0][featNo], startIndex+ features[0][featNo] ]
    outputFeatures = [] # create empty list to store race features in
    outputLog = [] # create empty list to store output log notes in
    previousKey = 'Top' # set prvious key to 'Top' so first loop entered correctly
    for key in sorted(featureDict): # iterate over the order sorted keys
        try: # try to form a logical race (ie top, bottom, ect ect finish
            if featureDict[key][0] != 'Finish': # check if the feature type is a finish
                outputFeatures.append(featureDict[key]) # if it isn't add it to the feature list
                previousKey = key # set previous key to the current key for the next loop
            elif featureDict[previousKey][0] != 'Finish': # if the current feature is a finish and the previous feature wasn't a finish (filters out multiple finishes)
                outputFeatures.append(featureDict[key]) # append the current feature
                outputLog.append("raceIdentified") # update the log
                previousKey = key # set previous key to the current key for the next loop

        except: # if an exception is raced it is likely due to the features not being in a logical order
            outputFeatures.append([]) # append no features
            outputLog.append("orderFailed") # append notes to log
            break # break out of the loop

    return outputFeatures, outputLog

def find_events(startDict, rawLog):
	
	"""
	finds_events takes the raw log and a dictionary with all start properties.
	
	"""
    import numpy as np

    raceEvents = {} # create empty dictionary to store race events in
    eventLog = {} # create empty dictionary to record program errors in
    marksDict = {} # crearte dictionary to record marks in

    for startNo in startDict.keys(): # iterate over each start and find the top and bottom marks
        raceLengths = [] # create empty data frame to store race lengths to trim log in last race
        startIndex = startDict[startNo][0] # set start index as the index at the start of the race
        if (startNo + 1) > max(startDict.keys()): # check if last start for day
            ## If start is last race of day follow logic to determine how much of the remaining log should be retained
            if raceLengths: # if race lengths is not empty
                timeStep = int(max(raceLengths)*1.25) # set the timestep
                if startDict[startNo][0]+timeStep > len(rawLog): # check last start plus time step is not longer than the raw log
                    endIndex = int(len(rawLog)-60) # if it is set the end index as 60 seconds before the end of the log
                else: # if timestep falls inside log
                    endIndex = int(startDict[startNo][0]+timeStep) # end index is 25# longer than the longest previous race

            else: # if no previous race lengths recorded
                endOptions = [startDict[startNo][0]+7200, len(rawLog)-60] # either two hours after the start or the end of the log will be the end index
                endIndex = int(min(endOptions)) # Choose the shortest option and set as end index

        else: # if not last race of day
            ## if start is not the last race of the day simply cut log at the following start
            endIndex = startDict[startNo+1][0]
            raceLengths.append(startDict[startNo+1][0]-startDict[startNo][0])

        raceFrame = rawLog.iloc[startDict[startNo][0]:endIndex] # create data frame of all lines between the last start and the end index
				
				###try except statement here if marks dont exist
        refLat = (startDict[startNo][2]+ startDict[startNo][4])/2 # find the average Lat between the pin and boat to use as reference point for distance calc
        refLon = (startDict[startNo][3]+ startDict[startNo][5])/2 # find the average Lon between the pin and boat to use as reference point for distance calc
        raceFrame = raceFrame.assign(Dist = list(np.square(raceFrame['Lat']-refLat) + np.square(raceFrame['Lon']-refLon)))  # calculate the distance from the ref point (midline) to the boats current location

        raceEvents[startNo], eventLog[startNo] = FindFeature(raceFrame, startIndex, startNo)

    marks = []
    for event in raceEvents[startNo]:
        if len(event) > 0:
            try:
                logLine = raceFrame.iloc[event[1]-60] # return the line of the log file 1min. before the mark rounding
                lat = logLine['MkLat'] # record the active mark Lat
                lon = logLine['MkLon'] # record the active mark Lon

            except:
                logLine = raceFrame.iloc[event[1]] # if finding an active mark fails, take the mark location as the location of the boat at the race event
                lat = logLine['Lat'] # record the Lat
                lon = logLine['Lon'] # record the Lon

            marks.append([lat,lon])

    marksDict[startNo] = marks

    raceIndex = {}
    marksDict = {}
    raceEvents = {}
    eventLog = {}
    for startNo in startDict.keys():
        raceLengths = [] # create empty data frame to store race lengths to trim log in last race
        startIndex = startDict[startNo][0] # set start index as the index at the start of the race
        if (startNo + 1) > max(startDict.keys()): # check if last start for day
            if raceLengths: # if race lengths is not empty
                timeStep = int(max(raceLengths)*1.25) # set the timestep
                if startDict[startNo][0]+timeStep > len(rawLog): # check last start plus time step is not longer than the raw log
                    endIndex = int(len(rawLog)-60) # if it is set the end index as 60 seconds before the end of the log
                else: # if timestep falls inside log
                    endIndex = int(startDict[startNo][0]+timeStep) # end index is 25# longer than the longest previous race

            else: # if no previous race lengths recorded
                endOptions = [startDict[startNo][0]+7200, len(rawLog)-60] # either two hours after the start or the end of the log will be the end index
                endIndex = int(min(endOptions)) # Choose the shortest option and set as end index

        else: # if not last race of day
            endIndex = startDict[startNo+1][0]
            raceLengths.append(startDict[startNo+1][0]-startDict[startNo][0])
        raceIndex[startNo] = [startIndex, endIndex]
        raceFrame = rawLog.iloc[startDict[startNo][0]:endIndex] # create data frame of all lines between the last start and the end index
        raceFrame = raceFrame.assign(Dist = list(np.square(raceFrame['Lat']-refLat) + np.square(raceFrame['Lon']-refLon)))  # calculate the distance from the ref point (midline) to the boats current location

        raceEvents[startNo], eventLog[startNo] = FindFeature(raceFrame, startIndex, startNo)

        if raceEvents[startNo][0]:

            marks = []
            for event in raceEvents[startNo]:

                try:
                    logLine = raceFrame.iloc[event[1]-60] # return the line of the log file 1min. before the mark rounding
                    lat = logLine['Mk Lat'] # record the active mark Lat
                    lon = logLine['Mk Lon'] # record the active mark Lon
                except:
                    logLine = raceFrame.iloc[event[1]] # if finding an active mark fails, take the mark location as the location of the boat at the race event
                    lat = logLine['Lat'] # record the Lat
                    lon = logLine['Lon'] # record the Lon

                marks.append([lat,lon])

            marksDict[startNo] = marks

    return     raceIndex, marksDict, raceEvents, eventLog

    ## Plot the track for each race for checking

def plot_race(startDict, raceEvents, rawLog, marksDict, raceIndex):
"""
Plotting
""" 
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    plots = {}

    for startNo in startDict.keys():
        if raceEvents[startNo][0]:
            plotFrame = rawLog.iloc[startDict[startNo][0]-360:raceEvents[startNo][-1][2]] # create a dataframe with race data to plot from 6min before the start to the finish
            leg = [] # create an empty list to store the leg id in
            eventNo = 0 # set the event to 0 (index)
            for i in range(len(plotFrame)): # iterate over the length of the plot frame
                if i < 361:
                    leg.append("PreStart") # first lines are prestart
                elif i <= raceEvents[startNo][eventNo][1]+361: # if the counter is less than the index of the next event record the leg number
                    leg.append("leg"+str(eventNo+1))
                else: # if not increase the event number
                    leg.append("leg"+str(eventNo+1))
                    eventNo += 1
            marks_array = np.array(marksDict[startNo])# create array of mark lats and longs for easy acces when plotting
            plotFrame = plotFrame.assign(Leg=leg) # add the Leg to the plot data frame
            plotFrame['Leg'] = plotFrame['Leg'].astype("category") # make leg column categorical for easy plotting
            fig = plt.figure()
            sns.scatterplot(data = plotFrame, x ='Lat', y = 'Lon', hue = 'Leg').set(title='Race # '+str(startNo))
            plt.plot([startDict[startNo][2], startDict[startNo][4]], [startDict[startNo][3], startDict[startNo][5]], linewidth = 2)
            plt.scatter(marks_array.transpose()[0],marks_array.transpose()[1], color = 'r')
            markCount = 1
            for mark in marks_array:
                plt.annotate("Mark "+str(markCount), mark)
                markCount += 1

        else:

            fig = plt.figure()
            plotFrame = rawLog.iloc[startDict[startNo][0]-360:raceIndex[startNo][1]]
            sns.scatterplot(data = plotFrame, x ='Lat', y = 'Lon').set(title='Race # '+str(startNo)+" "+str(startDict[startNo][-1]))
            plt.plot([startDict[startNo][2], startDict[startNo][4]], [startDict[startNo][3], startDict[startNo][5]], linewidth = 2)

        plots[startNo] = fig
        plt.close(fig)

    return plots

def sails_from_rrp(path):
	## move to rrp utils
	## ad support for staysails
	## sailtypes as defaults arguments
	## make sails and manual vars a class??
	"""
	sails_from_rrp imports a rrp config (or event file??) from a location defined by "path".
	The XML file is parsed and returns a dictionary containing all the individual sail codes for each sail type.
	"""
    import xml.etree.ElementTree as ET
		## import tree and get root
    tree = ET.parse(path)
    root = tree.getroot()
	
		## Empty lists for each sail type, pluys final dictionary
    mains = []
    jibs = []
    spins = []
    stays = []
    sail_dict = {}
		
		## find each sail with the "sailgroup" label corepsonding to the sail type and appened to
		## the apropriate list and then add to dictionary
    for sail in root.findall("./sailinventory/item[@sailgroup='M']"):
        mains.append(sail.get('name'))
    sail_dict['mains'] = mains

    for sail in root.findall("./sailinventory/item[@sailgroup='H']"):
        jibs.append(sail.get('name'))
    sail_dict['jibs'] = jibs

    for sail in root.findall("./sailinventory/item[@sailgroup='S']"):
        spins.append(sail.get('name'))
    sail_dict['spins'] = spins

		## make jibs available as staysails
    stays = jibs
    stays = ['']+stays
    sail_dict['stays'] = stays


    return sail_dict

def manualVars_from_rrp(path):
	"""
	manualVars_from_rrp inspects a rrp config file for all the manual entry variables defined.
	It returns the variable names, types and categories (for categorical variables)
	"""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    categorical = {}
    integer = []
    floatType = []
    for var in root.findall("./perfvariables/item[@calculationtype='ManualEntry']"):
        if var.get('type') == 'FromList':
            categorical[var.get('name')] = var.get('calculationformula')

        if var.get('type') == 'Scalar':
            if var.get('decimals') == '0':
                integer.append(var.get('name'))

            else:
                floatType.append((var.get('name')))
                

    categorical = {k:v.split(';') for (k,v) in categorical.items()}

    return categorical, integer, floatType

def build_var_dict(sail_dict, categorical, integer, floatType):
	## move to a eventField utils file
	
	"""
	build_var_dict takes all inputed variable dictionaries and combines into a single dictionary formatted as follows (to suit build selector race)
	
	{Sails: {mains: [list of mainsails],
					{jibs: [List of jibs],
					......},
	categorical: {categorical_1: [List of values],
								categorical_2:[List of values],
								....},
	integer: [List of integer variables],
	
	float: [list of float variables]}
	"""
    variables_dict = {'sails':sail_dict,
                      'categorical': categorical,
                      'integer': integer,
                      'floatType': floatType}

    return variables_dict

def get_rrp_config(path, use_config):
    ## move to eventfiled utils
    """
    get_rrp_config is a helper function that in future will allow users to input their own sail lists and variables.
    
    It returns a dictionary of all the manual entry variables and sails
    """

    if use_config:
        sail_dict = sails_from_rrp(path)
        categorical, integer, floatType = manualVars_from_rrp(path)


    else:
        sail_dict = {'mains':['Mainsail'],
                     'jibs':['Jib'],
                     'spins':['Spinaker'],
                     'stays':['Staysail']}
        categorical = {}
        integer = {}
        floatType = {}

    var_dict = build_var_dict(sail_dict,categorical, integer, floatType)

    return var_dict

def build_selectors_race(variables_dict):
    ## possibly better as a class
    
    """
    build_selectors_race take a dictionary of variables (in sails) strucured as follows
    
    {Sails: {mains: [list of mainsails],
    {jibs: [List of jibs],
    ......},
    categorical: {categorical_1: [List of values],
    categorical_2:[List of values],
    ...},
    integer: [List of integer variables],

    float: [list of float variables]}

    The apropriate widget type is then created for each variable / sail selector
    
    These are then arranged into horizontal and vertical boxes

    The sail widgets are grouped horizontally across.
    The manual variable widgets are grouped vertically as categorical, integer and float.
    The 3 vertical boxes are then group horizontally.
    Finally the sail widgets are stacked ontop of the manual variable widgets.

    Then all the individual widgets and the parent formatted widget are returned as a dictionary

    """
    import ipywidgets as widgets
    
    ## create individual dictionaries for sails and each variable 
    sail_dict = variables_dict['sails']
    categorical = variables_dict['categorical']
    integer = variables_dict['integer']
    floatType = variables_dict['floatType']

    ## create Select type widgets for the sails
    sail_selectors = {}
    for sailType in sail_dict.keys():
        sail_selectors[sailType] = widgets.Select(options=sail_dict[sailType],
                                              description=sailType)
    ## create Dropdown type widgets for all categorical variables
    categorical_selectors = {}
    for catVariable in categorical.keys():
        categorical_selectors[catVariable] = widgets.Dropdown(options=categorical[catVariable],
                                                          description=catVariable)
    ## arrange all categorical Dropdown widgets vertically in a box
    left_box = widgets.VBox([item for item in categorical_selectors.values()])
    
    ## integer variable widgets are crerated using IntText
    int_selectors = {}
    for intVariable in integer:
        int_selectors[intVariable] = widgets.IntText(description=intVariable)

    ## the middle manual variable are grouped vertically in a box
    mid_box = widgets.VBox([item for item in int_selectors.values()])

    ## the float selectors widgets are created using the FloatText widget
    float_selectors = {}
    for floatVariable in floatType:
        float_selectors[floatVariable] = widgets.FloatText(description=floatVariable)

    ## the right manual variables are grouped together
    right_box = widgets.VBox([widget for widget in float_selectors.values()])

    ## All 3 manual entry variable boxes are grouped together horizontally
    bottom_box = widgets.HBox([left_box, mid_box, right_box])

    ## All sail selector widgets are grouped together horizontally 
    top_box = widgets.HBox([widget for widget in sail_selectors.values()])

    ## the 2 horizontal boxes are stacked ontop of eachother to create the layout for all race widgets.
    race_tab = widgets.VBox([top_box, bottom_box])

    selectors_dict = {'sails':sail_selectors,
                  'categorical': categorical_selectors,
                  'integer': int_selectors,
                  'floatType': float_selectors,
                  'formatted': race_tab}

    return selectors_dict

def build_selectors_day(raceCount, variables_dict):
    """
    Racecount = integer # of races
    variables Dict = dictionary of sails and manual variables.
    
    Build_selectors_day essentially just calls "build_selector_dict_race" for each race and combines this into a tab per race.
    
    Directional links are also generated so as changes made in a given race are pushed to all future races but not updated for previous races.

    All widgets and the formatted widgets and directional links are then returned as a dictionary

    
    """
    import ipywidgets as widgets

    ## iterate over each start, generate widgets and make tabs
    selectors_dict_day = {}
    for i in range(raceCount):
        selectors_dict_day[i+1] = build_selectors_race(variables_dict) # use build_race_selector to build the selectors for each race

    race_numbers = [int for int in range(1,raceCount+1)] # create list of race numbers
    race_tabs = [selectors_dict_day[i]['formatted'] for i in race_numbers] # create a tab for each race
    tab_names = ['Race '+str(i) for i in race_numbers] # create list of tb names
    selector_tabs = widgets.Tab() # generate tabs
    selector_tabs.children = race_tabs # assign content to tabs
    [selector_tabs.set_title(i, title) for i, title in enumerate(tab_names)] # set tab titles

    selectors_dict_day['formatted'] = selector_tabs # add the formatted tabs to the dictionary for use later

## Generate directional links between tabs
    directional_links_dict = {}
    for race in list(selectors_dict_day.keys())[:-2]:
        directional_links_dict[race]={}
        for type in list(selectors_dict_day[race].keys())[:-1]:
            directional_links_dict[race][type]={}
            for var in selectors_dict_day[race][type].keys():
                directional_links_dict[race][type][var]= widgets.dlink((selectors_dict_day[race][type][var],'value'),(selectors_dict_day[race+1][type][var],'value'))

    selectors_dict_day['directional_link'] = directional_links_dict

    return selectors_dict_day

def load_and_process(log_path, config_path,use_config):
    ## move to eventfield utils file
    """
    loand_and_process is a simple worker function that executes the eventfield math and generates the selector widgets.
    """
    var_dict=get_rrp_config(config_path,use_config)
    log = PrepLog(log_path)
    startDict = find_starts(log) # obtain dictionary with start details
    raceIndex, marksDict, raceEvents, eventLog = find_events(startDict, log) # obtain race and event details
    selector_dict_day = build_selectors_day(len(startDict), var_dict)

    return log, var_dict, startDict, raceIndex, marksDict, raceEvents, eventLog, selector_dict_day

def update_default_chooser_path(chooser):
    ## move to eventfield utils
    
    if chooser.selected:
        chooser.default_path = chooser.selected_path
        chooser.title = chooser.title



        
def build_top_pane():
    ## rename as not top pane any more
    
    """
    Function builds the top pane by horizontally stacking the file choosers for the log file and the config file
    """
    from ipyfilechooser import FileChooser
    import ipywidgets as widgets

    log_file_chooser = FileChooser()
    log_file_chooser.title = 'Select log file'
    log_file_chooser.register_callback(update_default_chooser_path)

    config_file_chooser = FileChooser()
    config_file_chooser.title = 'Select config file'
    config_file_chooser.register_callback(update_default_chooser_path)

    checkbox_dont_load_config = widgets.Checkbox(value=True,
                                                 description='Use Sails and Variables from RRP Config')

    selector_box = widgets.HBox([log_file_chooser,config_file_chooser])
    buttons_box = widgets.HBox([checkbox_dont_load_config])
    top_pane = widgets.VBox([selector_box,buttons_box])

    top_pane_dict = {'file_choosers': [log_file_chooser, config_file_chooser],
                     'buttons': [checkbox_dont_load_config],
                     'formatted': [top_pane]}

    return top_pane_dict

def mid_pane_handler(top_pane_dict, loaded):
## this needs to be updated to show error log when incorrect files are loaded

    """
    Handler function that either retunrs a label widget showing "please load files" until the log and config are loaded and processed. Once this happens the plots and race tabs are displayed 
    """

    import ipywidgets as widgets

    ## if loaded boolean is False displaye msg
    if not loaded:
        not_loaded = [widgets.Label('Please Load Files')]
        mid_pane = widgets.Box(not_loaded)
        selector_dict_day = []
        raceEvents = []
        startDict = []
        log = []
    
    ## loaded boolean is True call load_and_process worker and create the middle pane as a vertical box with the plots tab and selectors tabs stacked ontop of eachother
    if loaded:
        selector_dict_day, plot_tab, log, raceEvents, marksDict, startDict = functions.load_and_process_worker(top_pane_dict)
        
    mid_pane = widgets.VBox([plot_tab,selector_dict_day['formatted']])
        
    display(mid_pane)

    
    return selector_dict_day, raceEvents, startDict, log

def build_plot_tabs(plots):
    
    """
    function to build a tab for each race to be plot. Plots arte handed to this function from the plot generator function
    """
    import ipywidgets as widgets
    from IPython.display import display
    
    plots_tab_dict = {}
    for start in plots.keys():
        plots_tab_dict[start] = widgets.Output()

    for start in plots_tab_dict.keys():
        with plots_tab_dict[start]:
            display(plots[start])

    plot_tab = widgets.Tab(children=list(plots_tab_dict.values()))

    for start in plots_tab_dict.keys():
        plot_tab.set_title(int(start-1), "Start #"+str(start))

    return plot_tab

def load_and_process_worker(top_pane_dict):
    ## move to event app utils
    """
    Worker function to take the inputs from the top tab and call the load and process and plot functions.
    The function returns the selectors, plot tabs aswell as the event and mark dicts.
    """
    use_config = top_pane_dict['buttons'][1].value
    log_path = top_pane_dict['file_choosers'][0].selected
    config_path = top_pane_dict['file_choosers'][1].selected

    log, var_dict, startDict, raceIndex, marksDict, raceEvents, eventLog, selector_dict_day = load_and_process(log_path,config_path,use_config)

    plots = plot_race(startDict, raceEvents, log, marksDict, raceIndex) # create plots for each race
    
    plot_tab = build_plot_tabs(plots)

    return selector_dict_day, plot_tab, log, raceEvents, marksDict, startDict

def get_event_values(selector_dict_day):
    event_values = {}
    for race in list(selector_dict_day.keys())[0:-2]:
        event_values[race] = {'manual_entry': {},
                              'upwind_sails': {},
                              'downwind_sails': {}}
        for manual_var_type in ['categorical', 'integer', 'floatType']:
            for manual_var in selector_dict_day[1][manual_var_type].keys():
                event_values[race]['manual_entry'][selector_dict_day[race][manual_var_type][manual_var].description] = selector_dict_day[1][manual_var_type][manual_var].value

        event_values[race]['upwind_sails'] = [selector_dict_day[race]['sails']['mains'].value, selector_dict_day[race]['sails']['jibs'].value, '' ,'' ]
        event_values[race]['downwind_sails'] = [selector_dict_day[race]['sails']['mains'].value,'', selector_dict_day[race]['sails']['stays'].value, selector_dict_day[race]['sails']['spins'].value ]
        event_values[race]['no_sails'] = [selector_dict_day[race]['sails']['mains'].value,'','', '']

    return event_values

def tidy_last_race_event(raceEvents):
    for race in raceEvents.keys():
        if len(raceEvents[race][0]):
            raceEvents[race][-1][0] != 'Finish'
            raceEvents[race][-1][0] = 'Finish'

    return raceEvents

def race_events_for_xml(raceEvents, event_values, log):
    """
    takes the race events dict and event values and creates the mark rounding and sails up events for during the race. These events are returned as a list of lists
    """

    events_for_XML = []

    for race in raceEvents.keys():
        mark_no = 1
        if len(raceEvents[race][0]) > 0:
            for event in raceEvents[race]:
                date = str(log['TimeStamp'][event[2]].date())
                time = str(log['TimeStamp'][event[2]].time())
                if event[0] == 'Top':
                    sails = event_values[race]['downwind_sails']
                    sails = sails[0]+";"+sails[1]+";"+sails[2]+";"+sails[3]
                    event_sails = [date, time, 'SailsUp', sails]
                    event_mark = [date, time, 'RaceMark', mark_no]
                    mark_no += 1

                if event[0] == 'Bottom':
                    sails = event_values[race]['upwind_sails']
                    sails = sails[0]+";"+sails[1]+";"+sails[2]+";"+sails[3]
                    event_sails = [date, time, 'SailsUp', sails]
                    event_mark = [date, time, 'RaceMark', mark_no]
                    mark_no += 1

                if event[0] == 'Finish':
                    sails = event_values[race]['no_sails']
                    sails = sails[0]+";"+sails[1]+";"+sails[2]+";"+sails[3]
                    event_sails = [date, time, 'SailsUp', sails]
                    event_mark = [date, time, 'RaceFinish', race]
                    mark_no = 1


                events_for_XML.append(event_mark)
                events_for_XML.append(event_sails)

    return events_for_XML

def start_events_for_xml(startDict, log, event_values):
    """
    Builds a list of lists containg all the start events plus prestart sails up and manual entry variables.
    """
    import pandas as pd
    events_for_XML = []
    for start in startDict.keys():
        date = str(log['TimeStamp'][startDict[start][0]].date())
        time = str((log['TimeStamp'][startDict[start][0]]-pd.Timedelta(minutes=6)).time())
        sails = event_values[start]['upwind_sails']
        sails = sails[0]+";"+sails[1]+";"+sails[2]+";"+sails[3]
        event_sails = [date, time, 'SailsUp', sails]
        time = str((log['TimeStamp'][startDict[start][0]]-pd.Timedelta(minutes=8)).time())
        event_manual = ""
        for var in event_values[start]['manual_entry'].keys():
            if var != list(event_values[start]['manual_entry'].keys())[-1]:
                event_manual = event_manual+var+"="+str(event_values[start]['manual_entry'][var])+";"
            else:
                event_manual = event_manual+var+"="+str(event_values[start]['manual_entry'][var])

        event_manual = [date, time, 'ManualEntry', event_manual]

        time = str(log['TimeStamp'][startDict[start][0]].time())
        event_race = [date, time, 'RaceStartGun', start]

        events_for_XML.append(event_sails)
        events_for_XML.append(event_manual)
        events_for_XML.append(event_race)

    return events_for_XML

def events_for_xml(raceEvents, startDict, event_values, log):
    ## move to eventField utils
    """
    events_for_xml is a simple worker function which calls a serries of functions to generate all the events for the days sailing and returns them as a single list ready for exporting to HTML
    """
    raceEvents = tidy_last_race_event(raceEvents)

    events_from_race = race_events_for_xml(raceEvents, event_values, log)

    events_from_start = start_events_for_xml(startDict, log, event_values)

    events_list = events_from_race + events_from_start
    
    log_timeSerries = log['TimeStamp'].dropna()
    start = log['TimeStamp'].dropna()[1]
    stop = log['TimeStamp'].dropna()[len(log_timeSerries)]
    
    events_list = events_list + [[str(start.date()),str(start.time()),"DayStart",""]] + [[str(stop.date()),str(stop.time()),"DayStop",""]]
    
    return events_list

def write_xml(events_list, filePath_export, fileName_export, boatName):
    ## rrp utils
    """
    write_xml takes a list of events and writes an XML file to suit SP RRP. The XML file is exported directly to the specified path
    """
    events_list.sort(key=lambda x: x[1])
    import xml.etree.ElementTree as ET
    export_root = ET.Element("daysail")
    
    boat = ET.Element("boat")
    boat.set("val", boatName)
    export_root.append(boat)
    
    date = ET.Element("date")
    date.set("val", events_list[0][0])
    export_root.append(date)
    
    events = ET.Element("events")
    export_root.append(events)


    for i in range(len(events_list)):
        event = ET.SubElement(events,"event")
        event.set("date",events_list[i][0])
        event.set("time", events_list[i][1])
        event.set("type", events_list[i][2])
        event.set("attribute", str(events_list[i][3]))
        event.set("comments","")
        event.set("labelalign","Top")

    export_tree = ET.ElementTree(export_root)

    with open (str(fileName_export)+".xml", "wb") as files :
        export_tree.write(files)
    
    print('Event File exported as '+str(fileName_export))

def export_to_xml_worker(selector_dict_day, raceEvents, startDict, log, filePath_export, fileName_export, boatName):
    ## move to eventField utils
    event_values = get_event_values(selector_dict_day)

    events_list_for_xml = events_for_xml(raceEvents, startDict, event_values, log)

    write_xml(events_list_for_xml, filePath_export, fileName_export, boatName)
    
    return events_list_for_xml
