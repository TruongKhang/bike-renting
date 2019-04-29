import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
import datetime

def read_file_stations(file_name):
    stations = pd.read_csv(file_name)
    stations = stations[~stations['Id'].isin([23, 25, 49, 69, 72])] # remove the old stations which are moved to 85, 86, 87, 88, 89
    stations['City'].replace(['San Francisco', 'Redwood City', 'Palo Alto', 'Mountain View', 'San Jose'], [94107, 94063, 94301, 94041, 95113], inplace=True)
    stations.rename(columns={'Id': 'id_station', 'City': 'Zip'}, inplace=True) # rename some columns
    stations.drop('Name', axis=1, inplace=True) # remove this column because it does not have any meaning
    return stations

def read_file_weathers(file_name):
    weather = pd.read_csv(file_name)
    weather['Date'] = pd.to_datetime(weather['Date'], format='%d/%m/%Y')
    list_date = weather['Date'].dt.date
    min_date, max_date = list_date.min(), list_date.max()
    num_days = (max_date - min_date).days + 1
    map_date2id = {(min_date+ datetime.timedelta(days=idx)): idx for idx in range(num_days)}
    weather['id_date'] = [map_date2id[date] for date in list_date]

    # preprocess the weather data
    # drop column Max Gust SpeedMPH because of exsiting many NaN values
    weather.drop('Max Gust SpeedMPH', axis=1, inplace=True)
    # Column 'Events' includes categorical values, NaN value means the weather is normal, no rain, no fog...
    weather['Events'].fillna('Normal', inplace=True)

    # Fill NaN values of each reamaining columns
    for col_name in weather.columns:
        if col_name not in ['Date', 'id_date', 'Zip', 'Events']:
            ids_nan = weather[col_name].index[weather[col_name].apply(np.isnan)]
            for idx in ids_nan:
                date = weather['id_date'][idx]
                cities = weather[weather['id_date'] == date]
                weather.at[idx, col_name] = cities[col_name].dropna().mean()

    return map_date2id, weather

def read_file_trips(file_name):
    trips = pd.read_csv(file_name)
    # replace the old stations 23, 25, 49, 69, 72
    trips['Start Station'].replace([23, 25, 49, 69, 72], [85, 86, 87, 88, 89], inplace=True)
    trips['End Station'].replace([23, 25, 49, 69, 72], [85, 86, 87, 88, 89], inplace=True)
    print(len(trips['Start Station'].unique()), len(trips['End Station'].unique()))

    #convert to datetime
    trips['Start Date'] = pd.to_datetime(trips['Start Date'], format='%d/%m/%Y %H:%M')
    trips['End Date'] = pd.to_datetime(trips['End Date'], format='%d/%m/%Y %H:%M')
    # calculate duration of trips
    trips['Duration'] = trips['End Date'] - trips['Start Date']
    trips['Duration'] = trips['Duration'].map(lambda x: x.total_seconds()/3600)
    print(trips['Duration'].min(), trips['Duration'].max())

    trips = trips[trips['Duration'] < 1000] # remove outlier
    print('Percentile of trips < 1h: ', len(trips[trips['Duration'] <= 1])/len(trips)*100)
    # draw histogram of duration of trips
    durations = trips.copy()
    durations = durations[durations['Duration'] < 24]
    durations.hist(column='Duration', bins=24)

    print('Percentile of subscriber type which is subscriber: ', len(trips[trips['Subscriber Type'] == 'Subscriber'])/len(trips)*100)
    return trips

"""
    Create the lag feature
"""
def lag_feature(table, lag_hours, col_name, keys=['id_hour', 'id_station']):
    temp = table[keys + [col_name]]
    for lag_hour in lag_hours:
        shifted = temp.copy()
        shifted.columns = keys + [col_name + '_lag_%d' %lag_hour]
        shifted[keys[0]] += lag_hour
        table = table.merge(shifted, on=keys, how='left')
    return table

class Dataset(object):
    def __init__(self, file_stations, file_weathers, file_trips):
        """
            Load data from files as dataframes
        """
        self.stations = read_file_stations(file_stations)
        self.map_date2id, self.weathers = read_file_weathers(file_weathers)
        self.trips = read_file_trips(file_trips)

        self.num_stations = self.stations['id_station'].count()
        print('Number of stations: ', self.num_stations)
        self.station_ids = self.stations['id_station'].values
        self.sid2id = {self.station_ids[i]: i for i in range(self.num_stations)}

        self.df_station_hour = None
        self.df_all = None

    def convert_data_trips(self):
        """
            Convert trips to a dataframe which contains information about 
            number of taken bikes (trips started), number of returned bikes (trips ended), net rate at each station every hour and everyday
        """
        
        min_date = self.trips['Start Date'].min()
        min_date = min_date.replace(minute=0, second=0)
        max_date = self.trips['End Date'].max()
        max_date = max_date.replace(minute=0, second=0)
        time_delta = max_date - min_date
        total_hours = int(time_delta.total_seconds() / 3600) + 1
        print('Total hours from ', min_date, ' to ', max_date, ': %d' %total_hours)

        # create matrix start station, end station
        start_matrix = dok_matrix((self.num_stations, total_hours))
        end_matrix = dok_matrix((self.num_stations, total_hours))

        for idx, row in self.trips.iterrows():
            # element of start matrix
            start_station_id = row['Start Station']
            start_date = row['Start Date'].replace(minute=0, second=0)
            delta = start_date - min_date
            start_id_col = int(delta.total_seconds() / 3600)
            start_id_row = self.sid2id[start_station_id]
            start_matrix[start_id_row, start_id_col] += 1
    
            # element of end matrix
            end_station_id = row['End Station']
            end_date = row['End Date'].replace(minute=0, second=0)
            delta = end_date - min_date
            end_id_col = int(delta.total_seconds() / 3600)
            end_id_row = self.sid2id[end_station_id]
            end_matrix[end_id_row, end_id_col] += 1
    
        start_keys = list(start_matrix.keys()) # list of pair (id_station, id_hour) which the bikes were taken in
        end_keys = list(end_matrix.keys()) # list of pair (id_station, id_hour) which the bikes were returned in
        total_keys = set([(row, col) for row in range(self.num_stations) for col in range(total_hours)]) # contains (id_station, id_hour) with zero bikes taken or returned
        positive_output_keys = set(start_keys + end_keys) # list of (id_station, id_hour) in which the bikes were taken or returned, consider as positive samples
        negative_output_keys = total_keys - positive_output_keys # negative samples
        positive_output_keys = list(positive_output_keys)
        negative_output_keys = list(negative_output_keys)

        # output considered as predicted variables, each element is net rate in (id_station, id_hour)
        output = end_matrix - start_matrix
        print(len(positive_output_keys))
        print(len(output.nonzero()[0]))

        # construct table of predicted values
        map_idhour2date = dict()
        for id_hour in range(total_hours):
            map_idhour2date[id_hour] = min_date + datetime.timedelta(hours=id_hour)
        postitive_stations_hour = list()
        negative_stations_hour = list()
        dict_col_nonzero_start = {row: start_matrix[row].nonzero()[1] for row in range(start_matrix.shape[0])}
        dict_col_nonzero_end = {row: end_matrix[row].nonzero()[1] for row in range(end_matrix.shape[0])}

        for keys, station_hour in [([positive_output_keys, postitive_stations_hour]), (negative_output_keys, negative_stations_hour)]:
            print(len(keys))
            for row, col in keys:
                id_station = self.station_ids[row]
                date = map_idhour2date[col]
                id_date =self. map_date2id[date.date()] # index of date (from 01/09/2014 to 31/08/2015)
                hour = date.hour # hour in day
                dow = date.weekday() # day of week
                hours_since_last_start = 0 # number of hours since the last event which the bikes were taken
                hours_since_last_end = 0 # number of hours since the last event which the bikes were returned
                col_nonzero_start = dict_col_nonzero_start[row] #start_matrix[row].nonzero()
                if min(col_nonzero_start) < col:
                    idx = np.where(col_nonzero_start < col)[0][-1]
                    hours_since_last_start = col - col_nonzero_start[idx]

                col_nonzero_end = dict_col_nonzero_end[row] #end_matrix[row].nonzero()
                if min(col_nonzero_end) < col:
                    idx = np.where(col_nonzero_end < col)[0][-1]
                    hours_since_last_end = col - col_nonzero_end[idx]
    
                station_hour.append([id_station, id_date, dow, col, hour, start_matrix[row, col], hours_since_last_start, end_matrix[row, col], hours_since_last_end, output[row, col]])
        df_positive = pd.DataFrame(postitive_stations_hour, columns=['id_station', 'id_date', 'dow', 'id_hour', 'hour', 'num_trips_start', 'hours_since_last_start', 'num_trips_end', 'hours_since_last_end', 'net_rate'])
        df_negative = pd.DataFrame(negative_stations_hour, columns=['id_station', 'id_date', 'dow', 'id_hour', 'hour', 'num_trips_start', 'hours_since_last_start', 'num_trips_end', 'hours_since_last_end', 'net_rate'])

        # merge two dataframe
        self.df_station_hour = df_positive.append(df_negative)
    
    def ceate_table_all(self):
        self.df_all = self.df_station_hour.merge(self.stations, on=['id_station'], how='left')
        self.df_all = self.df_all.merge(self.weathers, on=['id_date', 'Zip'], how='left')
        print(self.df_all.info())

    def create_features_matrix(self, save=None):
        # First, get the values in the past of some features which can not be observed at the present: num_trips_start, num_trips_end
        print('Get the observed features')
        self.df_all = lag_feature(self.df_all, [1,2,3] + list(range(24, 24*7, 24)), 'num_trips_start')
        self.df_all = lag_feature(self.df_all, [1,2,3] + list(range(24, 24*7, 24)), 'num_trips_end')
        self.df_all.drop(['num_trips_start', 'num_trips_end'], axis=1, inplace=True)
        # Remove the first week of dataset because we considered the length of lag as 7 days (one week)
        self.df_all = self.df_all[self.df_all['id_date'] >= 7]
        # drop columns which are keys of tables
        self.df_all.drop(['id_station', 'id_date', 'Date'], axis=1, inplace=True)
        # Create one-hot encoding for categorical columns
        print('Create one-hot encoding for categorical columns')
        categorical_columns = ['hour', 'dow', 'CloudCover', 'Events', 'Zip']
        for cate_col in categorical_columns:
            self.df_all[cate_col] = pd.Categorical(self.df_all[cate_col])
            df_one_hot = pd.get_dummies(self.df_all[cate_col], prefix=cate_col)
            self.df_all = pd.concat([self.df_all, df_one_hot], axis=1)
        self.df_all.drop(categorical_columns, axis=1, inplace=True)
        if save is not None:
            self.df_all.to_pickle(save)
        print(self.df_all.info())

if __name__ == '__main__':
    dataset = Dataset('../data/station_data.csv', '../data/weather_data.csv', '../data/trip_data.csv')
    dataset.convert_data_trips()
    dataset.ceate_table_all()
    dataset.create_features_matrix(save='../data/matrix_features.pkl')

