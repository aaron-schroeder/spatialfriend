# spatialfriend

> Python library for calculating geospatial data from gps coordinates.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](http://img.shields.io/:license-mit-blue.svg)](http://badges.mit-license.org)

---

## Table of Contents

- [Motivation](#motivation)
- [The Elevation Profile Smoothing Algorithm](#the-elevation-profile-smoothing-algorithm)
- [Dependencies and Installation](#dependencies-and-installation)
- [Example](#example)
- [Project Status](#project-status)
- [References](#references)
- [Contact](#contact)
- [License](#license)

---

## Motivation

Determining one's elevation on Earth's surface has become a lot easier thanks
to high-accuracy consumer GPS products and digital elevation models (DEMs) of
Earth's topography. Still, there are errors in GPS location and in every Earth
surface model. When working with elevation and position time series, for example
calculating instantaneous slopes during a trail running workout, stricter 
requirements are placed on the data. Even with a perfectly accurate DEM,
inaccurate GPS data can yield totally unreasonable elevation profiles and path
slopes, documenting work or elevation gain that the runner did not actually do.
The same can be said for a perfectly accurate GPS trace on an inaccurate DEM.

The goal of this project is to take GPS data of all resolutions, and return
geospatial data and calculations that actually match the athlete's experience.
No more unreasonably steep slopes or noisy data in your elevation profile 
making running power calculations meaningless. No more adding to your
workout's distance because your GPS was drifting around while you were 
waiting at a stoplight. No more wondering if those elevation measurements
you read on GPS device or barometric altimeter are accurate. No more apples to
oranges data comparisons because of differences between devices or datasets. 

This package is all about being able to hit record on that device, head
out for your run/hike/bike ride, and forget about it. Bring that messy
activity file and we will process the data once it is all done.

---

## The Elevation Profile Smoothing Algorithm

---

## Dependencies and Installation

### Base Installation

[GeoPy](https://github.com/geopy/geopy),
[Google Maps](https://github.com/googlemaps/google-maps-services-python),
[NumPy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org/), 
and [SciPy](https://www.scipy.org/) are required for the base installation.

`pip install spatialfriend` to install.

### Extra: Elevaton values from `.img` files

In addition to providing access to the Google Maps Elevation query service,
spatialfriend allows querying of user-owned `.img` files that contain 
elevation data. Such files are available from 
[the National Map's download page](https://viewer.nationalmap.gov/basic/).

[GDAL](https://pypi.org/project/GDAL/) and [utm](https://github.com/Turbo87/utm)
are required for this extra feature.

`pip install spatialfriend[img]` to install.

### Extra: Elevation values from the National Map

spatialfriend allows querying of the National Map's 
[Elevation Point Query Service](https://nationalmap.gov/epqs/). This 
service exposes data from the National Map's 1/3 arc-second Digital 
Elevation Model. 1/3 arc-second refers to the data's horizontal 
resolution in terms of degrees; this equates to roughly 30 meters.

[requests](https://pypi.org/project/requests/) and 
[urllib3](https://github.com/urllib3/urllib3) are required for this 
extra feature.

`pip install spatialfriend[tnm]` to install.

---

## Example
```python
import spatialfriend as sf

import config  # a hidden file containing user-specific info.

# Initialize an Elevation object.
lonlat_list = [[-105.0, 40.0], [-105.1, 40.0], [-105.1, 40.1]]
elev_helper = sf.Elevation(lonlat_list,
                           user_gmaps_key=config.my_gmaps_key,
                           img_dir=config.my_img_dir)

# An array of cumulative distances to each point from the beginning
# of the lonlat sequence.
distances = elev_helper.distance

# Get google maps elevations at each point.
google_elevs = elev_helper.google(units='feet')

# Get elevations from the .img files that live in `img_dir`
# (if those img files cover the specified coordinates).
img_elevs = elev_helper.img(units='feet')

# Compare the elevation gain using the different elevation sources.
print(sf.elevation_gain(google_elevs))
print(sf.elevation_gain(img_elevs))

# Use the algorithm to smooth the elevation profiles, and calculate
# reasonable grades between points.
grade_google = sf.grade_smooth(distances, google_elevs)
grade_img = sf.grade_smooth(distances, img_elevs)
```

---

## Project Status

### Complete

- Create Python package.

- Implement an algorithm to smooth noisy elevation data.

- Implement a series of tests to ensure functionality as development progresses.

- Streamline input so user can be more hands-off.

### Current Activities

#### Documentation

- Describe the algorithms in more detail.

- Create a project wiki.

#### Benchmarking and Optimization

- Benchmark algorithm performance (speed, accuracy, and consistency):
   - Generate dummy series of (distance, elevation) data to check
     smoothing algorithm.
   - Generate series of GPS points to compare elevation datasets with
     and without smoothing.

### Future Work

- Implement an algorithm to smooth GPS position and speed data. 
  Most GPS-enabled activity trackers filter their speed and distance
  timeseries to remove measurement noise. I want to try and figure out
  how they do it, then replicate their techniques, and compare the
  smoothed position data.

- Make the elevation gain algorithm smarter, or create an alternate
  algorithm to emulate algorithms employed by Strava/TrainingPeaks/Garmin.

- Settle on an approach to querying the National Map.

---

## References

<!--
- [A Developer Diary](http://www.adeveloperdiary.com/data-science/machine-learning/implement-viterbi-algorithm-in-hidden-markov-model-using-python-and-r/) for helping understand the nuts and bolts of the Viterbi algorithm in Python.
-->

---

## Contact

Reach out to me at one of the following places!

- Website: <a href="https://trailzealot.com" target="_blank">trailzealot.com</a>
- LinkedIn: <a href="https://www.linkedin.com/in/aarondschroeder/" target="_blank">linkedin.com/in/aarondschroeder</a>
- Twitter: <a href="https://twitter.com/trailzealot" target="_blank">@trailzealot</a>
- Instagram: <a href="https://instagram.com/trailzealot" target="_blank">@trailzealot</a>
- GitHub: <a href="https://github.com/aaron-schroeder" target="_blank">github.com/aaron-schroeder</a>

---

## License

[![License](http://img.shields.io/:license-mit-blue.svg)](http://badges.mit-license.org)

This project is licensed under the MIT License. See
[LICENSE](https://github.com/aaron-schroeder/spatialfriend/blob/master/LICENSE)
file for details.

---

<!--
## Features
## Documentation (Optional)
## Tests (Optional)
-->
