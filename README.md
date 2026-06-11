# TORCH Fire Detection Pipeline for NASA Space-to-Soil Challenge

Ground-based simulation of the TORCH onboard fire detection algorithm

TORCH (Thermal Optical Recognition and Communication Hardware) is a 3U CubeSat instrument designed to detect wildfires from low Earth orbit and deliver alerts in under 5 minutes via optical downlink. This repository simulates the onboard detection pipeline using real MODIS satellite data.

---

# Algorithm

The pipeline implements the two-stage detection approach described in the TORCH white paper:

# Stage 1 — LWIR temporal differencing**
Each swath's Band 20 brightness temperature (3.9 µm) is subtracted pixel-by-pixel from the previous swath. Pixels that heated rapidly between passes are flagged as candidates. This rejects slowly-varying backgrounds like diurnal terrain heating and coastal thermal gradients, which would overwhelm a static threshold.

# Stage 2 — SWIR cross-modal confirmation**
Candidate pixels are checked against Band 6 reflectance (1.64 µm). Both channels must confirm before a detection is recorded. Sun glint, industrial heat sources, and detector transients are hot in LWIR but do not produce simultaneous SWIR fire radiative power signatures.

# Post-processing
Connected components smaller than 4 pixels are discarded as noise. Confirmed detections are packaged as event records containing a pixel bounding box, pixel count, and a fire radiative power proxy — mirroring the alert packet TORCH would transmit to the ground station via PULSE-A optical downlink.

The algorithm adapts NASA's MODIS/VIIRS contextual fire detection approach (Justice et al. 2002, Schroeder et al. 2014) for onboard execution rather than ground processing.

---

# Data

The pipeline uses **MODIS MOD021KM** (Terra Level 1B, 1 km resolution) HDF4 files. Each file is one swath — a ~2030 × 1354 pixel strip (~2330 km wide) captured during a single satellite pass.

Consecutive swaths over the same region, separated by ~5 minutes, form a temporal pair for differencing. The pipeline processes all `.hdf` files in the `data/` folder in sorted (chronological) order.

# Download real fire data

Requires a free NASA EarthData account: [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/users/new)

```bash
# Download MODIS swaths for the 2020 Creek Fire (default)
python download_modis.py

# List all built-in fire events
python download_modis.py --list

# Download a specific named event
python download_modis.py --event dixie_fire

# Download for a custom location and date
python download_modis.py --lat 37.5 --lon -119.5 --date 2020-09-09
```

Built-in events include the Creek Fire (2020), Dixie Fire (2021), Bootleg Fire (2021), Camp Fire (2018), and Rim Fire (2013) — all large California/Oregon fires with confirmed MODIS detections.

---

# Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy pyhdf matplotlib earthaccess
```

---

# Usage

```bash
source venv/bin/activate
python src/firedetection.py
```

The pipeline loads all `.hdf` files from `data/`, runs detection, prints a summary, and displays a three-panel visualization per swath: brightness temperature, inter-frame delta, and confirmed fire pixels overlaid in orange.

**Example output (Creek Fire, September 9, 2020):**
```
Found 4 swath(s)

--- MOD021KM.A2020253.0525... ---   # 10:25 PM local, nighttime
  Method: single-frame spatial (no prior frame)
  No confirmed fire event.

--- MOD021KM.A2020253.0530... ---   # 10:30 PM local, nighttime
  Method: temporal differencing
  No confirmed fire event.          # SWIR inactive at night

--- MOD021KM.A2020253.1905... ---   # 12:05 PM local, daytime
  Method: temporal differencing
  Clean fire pixels : 106
  Fire clusters     : 17
  EVENT PACKET — 106 px, bbox rows (58, 2006), FRP proxy 1190998

--- MOD021KM.A2020253.1910... ---   # 12:10 PM local, daytime
  Method: temporal differencing
  Clean fire pixels : 140
  Fire clusters     : 21
  EVENT PACKET — 140 px, bbox rows (316, 2019), FRP proxy 285140

===== SUMMARY =====
Swaths processed : 4
Events confirmed : 2
```

---

## Repository Structure

```
wildfire_project/
├── src/
│   └── firedetection.py     # Detection pipeline
├── data/
│   └── *.hdf                # MODIS swaths (download separately)
├── download_modis.py        # NASA EarthData downloader
└── README.md
```

---

## References

- Justice et al. "An overview of MODIS Land data processing and product status." *Remote Sensing of Environment*, 2002.
- Schroeder et al. "The New VIIRS 375 m active fire detection data product." *Remote Sensing of Environment*, 2014.
- TORCH White Paper, University of Chicago Space Program, May 2026.
