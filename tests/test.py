import unittest
import datetime
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pandas
#from pandas.util.testing import assert_frame_equal, assert_series_equal

import spatialfriend as sf


class TestElevFuncs(unittest.TestCase):

  # Generate some dummy data, both as lists and series.
  distances = [0.0, 100.0, 200.0, 300.0]
  elevations = [0.0, 50.0, 75.0, 75.0]
  #distances = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
  #elevations =[0.0,  5.0, 10.0, 15.0, 17.5, 20.0, 22.5, 22.5, 22.5, 22.5]
  expected_grades = [np.nan, 0.5, 0.25, 0.0]
  dist_series = pandas.Series(distances)
  elev_series = pandas.Series(elevations)
  expected_array = np.array(expected_grades)
  expected_elevs_array = np.array(elevations)

  # Integration test: calculate grade from lists, series, and a mixture.
  grade_list_smooth = sf.grade_smooth(distances, elevations)
  grade_series_smooth = sf.grade_smooth(dist_series, elev_series)
  grade_mixed_1_smooth = sf.grade_smooth(dist_series, elevations)
  grade_mixed_2_smooth = sf.grade_smooth(distances, elev_series)
  grade_list_raw = sf.grade_raw(distances, elevations)
  grade_series_raw = sf.grade_raw(dist_series, elev_series)
  grade_mixed_1_raw = sf.grade_raw(dist_series, elevations)
  grade_mixed_2_raw = sf.grade_raw(distances, elev_series)

  # Integration test: calculate smooth elevs from lists, series, and a mixture.
  elev_list_smooth = sf.elevation_smooth(distances, elevations)
  elev_series_smooth = sf.elevation_smooth(dist_series, elev_series)
  elev_mixed_1_smooth = sf.elevation_smooth(dist_series, elevations)
  elev_mixed_2_smooth = sf.elevation_smooth(distances, elev_series)

  def test_raw_grade(self):
    assert_array_equal(self.grade_list_raw,
                       self.expected_array,
                       "Raw grades are not correct.")

  def test_smooth_grade(self):
    assert_allclose(self.grade_list_smooth,
                    self.expected_array,
                    atol=0.10,
                    err_msg="Smooth grades are not sufficiently close.")

  def test_raw_grade_type(self):
    self.assertIsInstance(self.grade_list_raw,
                          np.ndarray,
                          "Raw grades are not a ndarray.")

  def test_smooth_grade_type(self):
    self.assertIsInstance(self.grade_list_smooth,
                          np.ndarray,
                          "Smooth grades are not a ndarray.")

  def test_elevation_smooth(self):
    assert_allclose(self.elev_list_smooth,
                    self.expected_elevs_array,
                    atol=10.0,
                    err_msg="Smooth elevs are not sufficiently close.")

  def test_smooth_grade_type(self):
    self.assertIsInstance(self.elev_list_smooth,
                          np.ndarray,
                          "Smooth elevs are not a ndarray.")

class TestElevation(unittest.TestCase):

  # Integration test: create an Elevation from a list of coordinates
  # within Boulder County (the extent of the high-resolution DEM).
  latlon_list = [[-105.344016, 40.050938], 
                 [-105.344016, 40.025000],
                 [-105.260024, 40.025000], 
                 [-105.260024, 39.992579]]
  elevation = sf.Elevation(latlon_list)
  google_elevs = elevation.google
  lidar_elevs = elevation.lidar

  def test_create(self):
    self.assertIsInstance(self.elevation,
                          sf.Elevation,
                          "elevation is not an Elevation...")

  def test_google(self):
    self.assertIsInstance(self.google_elevs,
                          np.ndarray,
                          "Google elevs are not a ndarray.") 

  def test_lidar(self):
    self.assertIsInstance(self.lidar_elevs,
                          np.ndarray,
                          "Lidar elevs are not a ndarray.") 

  def test_methods_close(self):
    assert_allclose(self.lidar_elevs,
                    self.google_elevs,
                    atol=20.0,
                    err_msg="Google and lidar elevs are "  \
                            + "not sufficiently close.")


if __name__ == '__main__':
    unittest.main()
