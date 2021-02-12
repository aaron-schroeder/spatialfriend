# spatialfriend

> Python library for calculating geospatial data from gps coordinates.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](http://img.shields.io/:license-mit-blue.svg)](http://badges.mit-license.org)

---

## :zap: Project Update :zap: 

This project and its sister project [`heartandsole`](https://github.com/aaron-schroeder/heartandsole)
were my first real crack at activity file data analysis, and they do not reflect my
current standards. I am working on breaking out individual functionalities into 
separate packages. I will update the `README` of both packages to refer interested
users to my new self-contained, less messy projects. Stay tuned, and check out the 
[Project Status](#project-status) section below for specifics.

---

## Table of Contents

- [The Elevation Profile Smoothing Algorithm](#the-elevation-profile-smoothing-algorithm)
- [Dependencies and Installation](#dependencies-and-installation)
- [Example](#example)
- [Project Status](#project-status) <!-- - [References](#references) -->
- [Contact](#contact)
- [License](#license)

---

## The Elevation Profile Smoothing Algorithm

:tada: Moved over to my `py-elevation` package. 
[Read about it over there!](https://github.com/aaron-schroeder/py-elevation#the-elevation-smoothing-algorithm)

---

## Dependencies and Installation

### Base Installation

[GeoPy](https://github.com/geopy/geopy),
[Google Maps](https://github.com/googlemaps/google-maps-services-python),
[NumPy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org/), 
and [SciPy](https://www.scipy.org/) are required for the base installation.

`pip install spatialfriend` to install.

### Extra: Elevaton values from `.img` files

:zap: Update :zap: This is now handled by 
[my `elevation-query` package](https://github.com/aaron-schroeder/elevation-query#extra-elevation-values-from-img-and-geotiff-files),
and more flexibly too.

### Extra: Elevation values from the National Map

:zap: Update :zap: This is now handled by 
[my `elevation-query` package](https://github.com/aaron-schroeder/elevation-query#extra-elevation-values-from-img-and-geotiff-files),
and it works much faster over there.

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

- Create package on PyPi.
- Implement an algorithm to smooth noisy elevation data.
- Implement (some) tests.

#### Elevation and grade algorithms

- :tada: Publish a separate repo called [`py-elevation`](https://github.com/aaron-schroeder/py-elevation),
  which takes over for this repo and will be the location of continuing development.

#### Elevation data

- :tada: Publish a separate repo called [`py-elevationquery`](https://github.com/aaron-schroeder/elevation-query),
  which takes over the elevation-from-GPS role of this repo and will be the location of continuing development.

### Current Activities

- De-clutter this project, as it has splintered into many separate packages and projects. Refer users
  to my new projects that accomplish the functionality once found here.

### Future Work (likely in other repos)

#### Position and distance algorithms

- Publish a separate repo (tentatively named `py-distance`).
- Implement an algorithm to smooth GPS position and speed data. 
  Most GPS-enabled activity trackers filter their speed and distance
  timeseries to remove measurement noise. I want to try and figure out
  how they do it, then replicate their techniques, and compare the
  smoothed position data.

---

## Contact

Reach out to me at one of the following places!

<!-- - Website: <a href="https://trailzealot.com" target="_blank">trailzealot.com</a> -->
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
