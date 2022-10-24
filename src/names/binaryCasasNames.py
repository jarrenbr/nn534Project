from dataclasses import dataclass

#all binary
doorSensors = ['D021', 'D022', 'D023', 'D024',
       'D025', 'D026', 'D027', 'D028', 'D029', 'D030', 'D031', 'D032']

motionSensors = ['M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010',
       'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019',
       'M020']

allSensors = doorSensors + motionSensors

doorFalse = "CLOSE"
doorTrue = "OPEN"
motionFalse = "OFF"
motionTrue = "ON"

allActivities = ['Bathing', 'Bed_Toilet_Transition', 'Eating', 'Enter_Home', 'Housekeeping', 'Leave_Home',
                 'Meal_Preparation', 'Other_Activity', 'Personal_Hygiene', 'Relax', 'Sleeping_Not_in_Bed',
                 'Sleeping_in_Bed', 'Take_Medicine', 'Work']

sensColToOrd = { val : i for i, val in enumerate(allSensors)}

class ProcessedTime:
    _sin = "Sin"
    _cos = "Cos"
    _doy = "doy"
    _timeOfDay = "timeFromMidnight"
    doySin = _doy + _sin
    doyCos = _doy + _cos
    timeDaySin = _timeOfDay + _sin
    timeDayCos = _timeOfDay + _cos
    timeDif = "timeDif"
    order = [timeDif, doySin, doyCos, timeDaySin, timeDayCos]

@dataclass(frozen=True)
class rawLabels:
    time = "Time"
    sensor = "Sensor"
    signal = "Signal"
    activity = "Activity"
    correctOrder = [time, sensor, signal, activity]

rl = rawLabels


# features = ProcessedTime.order + [rl.signal] + allSensors
features = ProcessedTime.order + allSensors
nFeatures = len(features)

labels = allActivities
nLabels = len(allActivities)

allBinaryColumns = [rl.signal] + allSensors + allActivities

correctOrder = features# + labels
ganFeatures = correctOrder
nGanFeatures = len(ganFeatures)

class start_stop:
    def __init__(self, start, length):
        self.start = start
        self.stop = start + length

@dataclass(frozen=True)
class pivots:
    time = start_stop(0,len(ProcessedTime.order))
    signal = start_stop(time.stop, 0)
    sensors = start_stop(signal.stop, len(allSensors))
    activities = start_stop(sensors.stop, len(allActivities))
    features = start_stop(0, activities.stop)

colOrder = correctOrder
ordinalColDict = {i:c for i, c in enumerate(colOrder)}
colOrdinalDict = {c:i for i, c in enumerate(colOrder)}

@dataclass(frozen=True)
class house_names:
    allHomes = "All Real Home"
    synthetic = "Fake Home"
    home1 = "H1"
    home2 = "H2"
    home3 = "H3"

class house:
    def __init__(self, data, name, maxTimeDif=None):
        self.data = data
        self.name = name
        self.maxTimeDif = maxTimeDif