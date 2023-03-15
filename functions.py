def find_starts(rawLog):
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
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()

    mains = []
    jibs = []
    spins = []
    stays = []
    sail_dict = {}

    for sail in root.findall("./sailinventory/item[@sailgroup='M']"):
        mains.append(sail.get('name'))
    sail_dict['mains'] = mains

    for sail in root.findall("./sailinventory/item[@sailgroup='H']"):
        jibs.append(sail.get('name'))
    sail_dict['jibs'] = jibs

    for sail in root.findall("./sailinventory/item[@sailgroup='S']"):
        spins.append(sail.get('name'))
    sail_dict['spins'] = spins

    stays = jibs
    stays = ['']+stays
    sail_dict['stays'] = stays


    return sail_dict

def manualVars_from_rrp(path):
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
    variables_dict = {'sails':sail_dict,
                      'categorical': categorical,
                      'integer': integer,
                      'floatType': floatType}

    return variables_dict

def get_rrp_config(path, use_config):

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
    import ipywidgets as widgets

    sail_dict = variables_dict['sails']
    categorical = variables_dict['categorical']
    integer = variables_dict['integer']
    floatType = variables_dict['floatType']

    sail_selectors = {}
    for sailType in sail_dict.keys():
        sail_selectors[sailType] = widgets.Select(options=sail_dict[sailType],
                                              description=sailType)
    categorical_selectors = {}
    for catVariable in categorical.keys():
        categorical_selectors[catVariable] = widgets.Dropdown(options=categorical[catVariable],
                                                          description=catVariable)

    left_box = widgets.VBox([item for item in categorical_selectors.values()])

    int_selectors = {}
    for intVariable in integer:
        int_selectors[intVariable] = widgets.IntText(description=intVariable)

    mid_box = widgets.VBox([item for item in int_selectors.values()])

    float_selectors = {}
    for floatVariable in floatType:
        float_selectors[floatVariable] = widgets.FloatText(description=floatVariable)

    right_box = widgets.VBox([widget for widget in float_selectors.values()])

    bottom_box = widgets.HBox([left_box, mid_box, right_box])

    top_box = widgets.HBox([widget for widget in sail_selectors.values()])

    race_tab = widgets.VBox([top_box, bottom_box])

    selectors_dict = {'sails':sail_selectors,
                  'categorical': categorical_selectors,
                  'integer': int_selectors,
                  'floatType': float_selectors,
                  'formatted': race_tab}

    return selectors_dict

def build_selectors_day(raceCount, variables_dict):
    import ipywidgets as widgets

    selectors_dict_day = {}
    for i in range(raceCount):
        selectors_dict_day[i+1] = build_selectors_race(variables_dict)

    race_numbers = [int for int in range(1,raceCount+1)]
    race_tabs = [selectors_dict_day[i]['formatted'] for i in race_numbers]
    tab_names = ['Race '+str(i) for i in race_numbers]
    selector_tabs = widgets.Tab()
    selector_tabs.children = race_tabs
    [selector_tabs.set_title(i, title) for i, title in enumerate(tab_names)]

    selectors_dict_day['formatted'] = selector_tabs

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
    var_dict=get_rrp_config(config_path,use_config)
    log = PrepLog(log_path)
    startDict = find_starts(log) # obtain dictionary with start details
    raceIndex, marksDict, raceEvents, eventLog = find_events(startDict, log) # obtain race and event details
    selector_dict_day = build_selectors_day(len(startDict), var_dict)

    return log, var_dict, startDict, raceIndex, marksDict, raceEvents, eventLog, selector_dict_day

def update_default_chooser_path(chooser):
    if chooser.selected:
        chooser.default_path = chooser.selected_path
        chooser.title = chooser.title



        
def build_top_pane():
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
    import ipywidgets as widgets
    if not loaded:
        not_loaded = [widgets.Label('Please Load Files')]
        mid_pane = widgets.Box(not_loaded)
        selector_dict_day = []
        raceEvents = []
        startDict = []
        log = []
        
    if loaded:
        selector_dict_day, plot_tab, log, raceEvents, marksDict, startDict = functions.load_and_process_worker(top_pane_dict)
        mid_pane = widgets.VBox([plot_tab,selector_dict_day['formatted']])
        
    display(mid_pane)

    
    return selector_dict_day, raceEvents, startDict, log

def build_plot_tabs(plots):
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

    raceEvents = tidy_last_race_event(raceEvents)

    events_from_race = race_events_for_xml(raceEvents, event_values, log)

    events_from_start = start_events_for_xml(startDict, log, event_values)

    events_list = events_from_race + events_from_start


    return events_list

def write_xml(events_list, filePath_export, fileName_export):
    events_list.sort(key=lambda x: x[1])
    import xml.etree.ElementTree as ET
    export_root = ET.Element("daysail")
    
    
    date = ET.Element("date")
    date.set("val", events_list[0][0])
    export_root.append(date)
    
    events = ET.Element("events")
    export_root.append(events)

    event = ET.SubElement(events,"event")
    event.set("date",events_list[0][0])
    event.set("time", events_list[0][1])
    event.set("type", "DayStart")
    event.set("attribute", "")
    event.set("comments","")
    event.set("labelalign","Top")

    for i in range(len(events_list)):
        event = ET.SubElement(events,"event")
        event.set("date",events_list[i][0])
        event.set("time", events_list[i][1])
        event.set("type", events_list[i][2])
        event.set("attribute", str(events_list[i][3]))
        event.set("comments","")
        event.set("labelalign","Top")

    event = ET.SubElement(events,"event")
    event.set("date",events_list[0][0])
    event.set("time", events_list[len(events_list)][1])
    event.set("type", "DayStop")
    event.set("attribute", "")
    event.set("comments","")
    event.set("labelalign","Top")
    export_tree = ET.ElementTree(export_root)

    with open (str(fileName_export)+".xml", "wb") as files :
        export_tree.write(files)
    
    print('Event File exported as '+str(fileName_export))

def export_to_xml_worker(selector_dict_day, raceEvents, startDict, log, filePath_export, fileName_export):

    event_values = get_event_values(selector_dict_day)

    events_list_for_xml = events_for_xml(raceEvents, startDict, event_values, log)

    write_xml(events_list_for_xml, filePath_export, fileName_export)
    
    return events_list_for_xml
