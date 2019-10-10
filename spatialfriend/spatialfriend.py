import datetime
import math
import warnings

from geopy.distance import great_circle as distance
import googlemaps
import numpy as np
import pandas
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class Elevation(object):
  """Class to get elevations from lat-lon coordinates.

  Construction of an Elevation object reads in the latlon coords and
  calculates the cumulative distance to each point from the start 
  of the latlon sequence.

  TODO: Come up with a more descriptive class name.
  """

  def __init__(self, latlon_list, user_gmaps_key=None, img_dir=None):
    """Creates an Elevation from a list of latlon coords.

    Args:
      latlon_list: An array-like object of [lon, lat] pairs.
      user_gmaps_key: String representing Google Maps API key. Required
                      for 'google' method.
      img_dir: String representing the directory containing .img files 
               to be searched. Required for 'img' method.
    """
    self.user_gmaps_key = user_gmaps_key
    self.img_dir = img_dir

    if type(latlon_list) == pandas.DataFrame:
      latlon_list = latlon_list.values.squeeze()

    self.data = pandas.DataFrame(data=latlon_list,
                                 columns=['lon', 'lat'])

    self._clean_up_coordinates()

    # Build a new column for the DataFrame representing cumulative
    # distance to each point, with an initial zero value because 
    # no movement has occurred at first.
    distances_cum = [0.0]
    for i in range(1, len(self.data)):
      # Calculate cumulative distance up to this point by adding
      # the distance between the previous and current point 
      # to the cumulative distance to the previous point.
      row_prev = self.data.iloc[i-1]
      row_curr = self.data.iloc[i]
      distance_cum = distances_cum[i-1] +  \
          distance((row_prev['lat'], row_prev['lon']),
                   (row_curr['lat'], row_curr['lon'])).meters

      distances_cum.append(distance_cum)

    self.data['distance'] = distances_cum

  def _clean_up_coordinates(self):
    """Infers missing lat/lon coordinates in simple cases."""
    self.data.fillna(method='bfill', inplace=True)

  @property
  def latlons(self):
    return self.data[['lon', 'lat']]

  @property
  def distance(self):
    return np.array(self.data['distance']).squeeze()

  def google(self, units='meters'):
    """Queries google maps' Elevation API at each point.

    Args:
      units: String representing desired units. 'meters' or 'feet',
             case-insensitive.
    Returns:
      Google elevation values as a ndarray of floats, in meters.    
    """
    if self.user_gmaps_key is None:
      return None

    if 'google' not in self.data.columns:
      gmaps = googlemaps.Client(key=self.user_gmaps_key)

      # Google maps elevation api allows 500 elevation values
      # per request. Break latlon coordinates into 500-piece chunks
      # and pass to the api, then assemble returned elevations into one
      # consolidated list, and add to dataframe as a new column.
      unit_factor = 5280.0/1609 if units.lower() == 'feet' else 1.0
      elevs = []
      for _, chunk in self.latlons.groupby(np.arange(len(self.latlons)) // 500):

        # Format coordinates for google maps api request
        locations = [(float(row['lat']), float(row['lon'])) 
            for _, row in chunk.iterrows()]

        elevs.extend([round(elevobj['elevation'] * unit_factor, 1) 
            for elevobj in gmaps.elevation(locations)])
      self.data['google'] = elevs

    return np.array(self.data['google']).squeeze()

  def img(self, units='meters'):
    """Accesses elevation data from user-owned .img files.

    In my case, I have downloaded high-resolution elevation data for
    the Boulder area based on the National Map's 1-meter DEM.

    Args:
      units: String representing desired units. 'meters' or 'feet',
             case-insensitive.
    Returns:
      Elevations from files in .img directory as a ndarray of floats.
      NaN values are returned for coordinates outside the bounds of
      all the .img files. 
 
    TODO: Properly handle case where utm/gdal not installed, or
          the .img directory is not specified.
    """
    if self.img_dir is None:
      return None

    import glob
    import struct

    from osgeo import gdal
    import utm

    # TODO: Employ os.join so slashes aren't so important.
    file_objs = {}
    fnames = glob.glob(self.img_dir + '*.img')
    fnames.extend(glob.glob(self.img_dir + '*.IMG'))

    # Compile geospatial information about each IMG file, then close the
    # file to avoid blowing up the server's memory.
    for fname in fnames:
      img = gdal.Open(fname)
      xoffset, px_w, rot1, yoffset, rot2, px_h = img.GetGeoTransform()
      file_obj = {
        'xoffset': xoffset,
        'yoffset': yoffset,
        'px_w': px_w,
        'px_h': px_h,
        'points_to_try': []
      }
      file_objs[fname] = file_obj
      img = None

    # For each (lon, lat) pair, search through all img files for a
    # valid elevation coordinate, and save every one.
    for index, row in self.latlons.iterrows():
      posX, posY, zone_num, zone_letter = utm.from_latlon(row['lat'], row['lon'])

      # These values are not used in current implementation. They
      # facilitate finding the correct .IMG file by fname. Now however,
      # I search through every file.
      x_index = math.floor(posX / 10000)
      y_index = math.ceil(posY / 10000)

      for fname in fnames:
        file_obj = file_objs[fname]
        #px = int((posX - file_obj['xoffset']) / file_obj['px_w']) # x pixel
        #py = int((posY - file_obj['yoffset'])/ file_obj['px_h']) # y pixel
        
        # Find decimal pixel for x and y, then check if a valid index
        # exists by rounding each pixel index down or up. This means
        # there could be up to 4 different combinations of px/py.
        px_dec = (posX - file_obj['xoffset']) / file_obj['px_w']
        py_dec = (posY - file_obj['yoffset'])/ file_obj['px_h']
        pxs = [math.floor(px_dec), math.ceil(px_dec)]
        pys = [math.floor(py_dec), math.ceil(py_dec)]
   
        for px in pxs:
          for py in pys:
            if px > 0 and py > 0:
              file_obj['points_to_try'].append({'px': px,
                                                'py': py,
                                                'index': index})

    # For each IMG file, iterate through the pre-assembled list of
    # points which should have a valid elevation coordinate.
    # Catch exceptions resulting from each raster not having
    # the full spatial extent.
    unit_factor = 5280.0/1609 if units.lower() == 'feet' else 1.0
    elevs = np.full(len(self.latlons), np.nan)
    for fname in fnames:
      img = gdal.Open(fname)
      rb = img.GetRasterBand(1)

      # I think prevents printing to screen when point is outside grid.
      gdal.UseExceptions() 

      for point_to_try in file_objs[fname]['points_to_try']:
        try: 
          #rb = img.GetRasterBand(1)
          # Assumes 32 bit int aka 'float'.
          structval = rb.ReadRaster(point_to_try['px'],
                                    point_to_try['py'],
                                    1, 1, buf_type=gdal.GDT_Float32)
          intval = struct.unpack('f', structval)

          # Check if point within bounds, but elevation undefined (this
          # would likely be due to an incomplete image). The value used
          # for missing elevation data is a large negative number.
          if intval[0] > -9999:
            ix = int(point_to_try['index']);
            if elevs[ix] != intval[0]*unit_factor: 
              # There isn't a value yet.
              elevs[ix] = intval[0]*unit_factor
          #rb = None
        except:
          pass

      img = None
      rb = None

    # Clean up null elevation values in the array by forward-filling. 
    # Should be very few, right at IMG edges. This method should work
    # except in the case of leading nan values in the array.
    #ixs_missing = np.argwhere(np.isnan(elevs))
    #for ix_missing in ixs_missing:
    #  elevs[ix_missing] = elevs[ix_missing-1]

    # Round to the nearest tenth of a unit. Higher precision elevation
    # differences could not be differentiated from measurement noise.
    elevs_round = np.around(elevs, decimals=1)

    return elevs_round.squeeze()


def _retry(exceptions, tries=4, delay=3, backoff=2, logger=None):
  """A deco to keep the National Map's elevation query working."""
  from functools import wraps
  import time


  def deco_retry(f):
    @wraps(f)
    def f_retry(*args, **kwargs):
      mtries, mdelay = tries, delay
      while mtries > 1:
        try:
          return f(*args, **kwargs)
        except exceptions as e:
          msg = ('{}, Retrying in {} seconds...').format(e, mdelay)
          if logger:
            logger.warning(msg)
          else:
            print(msg)
          time.sleep(mdelay)
          mtries -= 1
          mdelay *= backoff
      return f(*args, **kwargs)
    return f_retry
  return deco_retry


@_retry(ValueError, tries=4, delay=3, backoff=2)
def _query_natmap(lat, lon, units='Meters'):
  """Returns elevation data with 1/3 arc-second resolution (~30m).

  Data is accessed via based the National Map's Elevation Point Query 
  Service. The 1/3 arc-second digital elevation model covers the
  entire US, so this should work anywhere.
 
  Because the Elevation Point Query Service only permits one point at
  a time, this function is not suitable for querying large sets of
  latlon coordinates. It goes incredibly slow.

  TODO: Find an approach that goes faster or lets this function work
        in the background. 
  """
  import json
  import ssl

  from urllib.request import urlopen

  ssl.match_hostname = lambda cert, hostname: True
  url = 'https://nationalmap.gov/epqs/pqs.php?x=' + str(lon)  \
        + '&y=' + str(lat) + '&units=' + units.capitalize() + '&output=json'
  request = urllib.request.urlopen(url)
  json_request = json.load(request)

  return json_request['USGS_Elevation_Point_Query_Service'][
                      'Elevation_Query']['Elevation']


def _query(latlon):
  import json
  import ssl

  from urllib.request import urlopen
  #import requests

  ssl.match_hostname = lambda cert, hostname: True
  url = 'https://nationalmap.gov/epqs/pqs.php?x=' + str(latlon[0])  \
        + '&y=' + str(latlon[1]) + '&units=Meters&output=json'
  #request = urllib.request.urlopen(url)
  request = urlopen(url)
  #request = requests.get(url)
  #return request.json()['USGS_Elevation_Point_Query_Service'][
  #                      'Elevation_Query']['Elevation']

  return json.load(request)['USGS_Elevation_Point_Query_Service'][
                            'Elevation_Query']['Elevation']


def _elevations_natmap(latlons, units='Meters'):
  from multiprocessing import Pool

  with Pool() as p:
    print(p.map(_query, latlons))


def elevation_gain(elevations):
  """Naive elevation gain calculation.

  TODO: Make this algorithm smarter so noise doesn't affect it as much.
  """
  return sum([max(elevations[i+1] - elevations[i], 0.0)
              for i in range(len(elevations) - 1)])


def elevation_smooth(distances, elevations, window_length=3, polyorder=2):
  """Smooths noisy elevation data for use in grade calculations.

  Because of GPS and DEM inaccuracy, elevation data is not smooth.
  Calculations involving terrain slope (the derivative of elevation
  with respect to distance, d_elevation/d_x) will not yield reasonable
  values unless the data is smoothed. 

  This method's approach follows the overview outlined in the 
  NREL paper found in the Resources section and cited in README.
  The noisy elevation data is downsampled and passed through a dual
  filter, consisting of a Savitzy-Golay (SG) filter and a binomial
  filter. Parameters for the filters were not described in the paper,
  so they must be tuned to yield intended results when applied
  to the data.

  Args:
    distances: Array-like object of cumulative distances along a path. 
    elevations: Array-like object of elevations above sea level
                  corresponding to the same path.
    window_length: An integer describing the length of the window used
                   in the SG filter. Must be positive odd integer.
    polyorder: An integer describing the order of the polynomial used
               in the SG filter, must be less than window_length.
    TODO(aschroeder) n: An integer describing the complexity of 
                        the binomial filter, must be 1 or greater.

  TODO(aschroeder) Combine a binomial filter with existing SG filter
                   and test effects on algorithm performance.
  """
  distances = pandas.Series(distances, name='distance')
  elevations = pandas.Series(elevations, name='elevation')

  # Subsample elevation data in evenly-spaced intervals, with each
  # point representing the median elevation value. If data is spaced
  # further than the subsampling interval, suppress warnings about 
  # taking the mean of empty slices and just interpolate between points.
  len_sample = 30.0  # meters
  n_sample = math.ceil(distances.iloc[-1] / len_sample)
  idx = pandas.cut(distances, n_sample)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    data_ds = elevations.groupby(idx).apply(np.median).interpolate(
        limit_direction='both').to_frame()
  data_ds['distance'] = pandas.IntervalIndex(
      data_ds.index.get_level_values('distance')).mid

  # Pass downsampled data through a Savitzky-Golay filter (attenuating
  # high-frequency noise). 
  # Then throw out any points where the elevation difference resulting
  # from filtration exceeds a threshold, and backfill via interpolation.
  # TODO (aschroeder): Add a second, binomial filter.
  # TODO (aschroeder): Fix the scipy/signal/arraytools warning!
  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    data_ds['sg'] = savgol_filter(
        data_ds['elevation'], window_length, polyorder)  
  elevation_diff = np.abs(data_ds['elevation'] 
                        - data_ds['sg'])
  data_ds['elevation'][elevation_diff > 5.0] = np.nan
  data_ds['interp'] = data_ds['elevation'].interpolate()

  # Pass backfilled elevation profile through the Savitzky-Golay
  # filter to eliminate noise from the distance derivative. 
  # Calculate elevations at the original distance values
  # via interpolation.
  # TODO (aschroeder): Add a second, binomial filter.
  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    data_ds['sg_final'] = savgol_filter(
        data_ds['interp'], window_length, polyorder)
  interp_function = interp1d(
      data_ds['distance'], data_ds['sg_final'], 
      fill_value='extrapolate', kind='linear')
      #fill_value='extrapolate', kind='quadratic')
  smooth = interp_function(distances)

  return smooth


def grade_smooth(distances, elevations):
  """Calculates smoothed point-to-point grades.

  TODO(aschroeder): check if distances and elevations are same length.
  Args:
    distances: Array-like object of cumulative distances along a path. 
    elevations: Array-like object of elevations above sea level
                corresponding to the same path.
  """
  distances = pandas.Series(distances).reset_index(drop=True)
  elevations = pandas.Series(elevations).reset_index(drop=True)
  elevations_smooth = pandas.Series(elevation_smooth(distances, elevations))
 
  grade = elevations_smooth.diff() / distances.diff()

  # Clean up spots with NaNs from dividing by zero distance.
  # This assumes the distances and elevations arrays have no NaNs.
  grade.fillna(0, inplace=True) 

  return np.array(grade)


def grade_raw(distances, elevations):
  """Calculates unsmoothed point-to-point grades.

  TODO(aschroeder): check if distances and elevations are same length.
  Args:
    distances: Array-like object of cumulative distances along a path. 
    elevations: Array-like object of elevations above sea level
                corresponding to the same path.
  """
  distances = pandas.Series(distances).reset_index(drop=True)
  elevations = pandas.Series(elevations).reset_index(drop=True)

  grade = elevations.diff() / distances.diff()

  # Clean up spots with NaNs from dividing by zero distance.
  # This assumes the distances and elevations arrays have no NaNs.
  grade.fillna(0, inplace=True) 

  return np.array(grade)
