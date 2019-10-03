# spatialfriend

> Python library for calculating geospatial data from gps coordinates.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](http://img.shields.io/:license-mit-blue.svg)](http://badges.mit-license.org)

<!--
[![Build Status](http://img.shields.io/travis/badges/badgerbadgerbadger.svg?style=flat-square)](https://travis-ci.org/badges/badgerbadgerbadger) [![Dependency Status](http://img.shields.io/gemnasium/badges/badgerbadgerbadger.svg?style=flat-square)](https://gemnasium.com/badges/badgerbadgerbadger) [![Coverage Status](http://img.shields.io/coveralls/badges/badgerbadgerbadger.svg?style=flat-square)](https://coveralls.io/r/badges/badgerbadgerbadger) [![Badges](http://img.shields.io/:badges-9/9-ff6799.svg?style=flat-square)](https://github.com/badges/badgerbadgerbadger)
-->


---

## Table of Contents

- [Motivation](#motivation)
- [The Elevation Profile Smoothing Algorithm](#the-elevation-profile-smoothing-algorithm)
- [Example](#example)
- [Project Status](#project-status)
- [Inspiration](#inspiration)
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

## Example
```python
import spatialfriend as sf

```

---

## Project Status

### Complete

- Create Python package.

- Implement a series of tests to ensure functionality as development progresses.

### Current Activities

- Streamline input so user can be more hands-off.

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

---

## Inspiration <!-- References ? -->

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

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2019 © <a href="https://trailzealot.com/about" target="_blank">Aaron Schroeder</a>.

<!--
## Installation

- All the `code` required to get started
- Images of what it should look like

### Clone

- Clone this repo to your local machine using `https://github.com/EricSchraider/mapmatching`

### Setup

- If you want more syntax highlighting, format your code like this:

> update and install this package first

```shell
$ brew update
$ brew install fvcproductions
```

> now install npm and bower packages

```shell
$ npm install
$ bower install
```

---
-->


<!--
## Features
## Documentation (Optional)
## Tests (Optional)
-->
