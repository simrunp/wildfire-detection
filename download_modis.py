"""
Download MODIS MOD021KM (Terra Level 1B) granules for known wildfire events.

Requires a free NASA EarthData account: https://urs.earthdata.nasa.gov/users/new
On first run you will be prompted for your username and password, which are
saved to ~/.netrc so you only need to do this once.

Usage:
    python download_modis.py                        # downloads default fire event
    python download_modis.py --event creek_fire     # specific event
    python download_modis.py --list                 # show all built-in events
    python download_modis.py --lat 37.5 --lon -119.5 --date 2020-09-09  # custom
"""

import argparse
import os
import sys
import earthaccess

# ---------------------------------------------------------------------------
# Known wildfire events: (date, center_lat, center_lon, description)
# These are well-documented fires with confirmed MODIS detections.
# ---------------------------------------------------------------------------
FIRE_EVENTS = {
    "creek_fire": {
        "date": "2020-09-09",
        "lat": 37.37,
        "lon": -119.27,
        "desc": "Creek Fire, Sierra National Forest, CA — largest single fire in CA history at time",
    },
    "dixie_fire": {
        "date": "2021-07-14",
        "lat": 40.08,
        "lon": -121.26,
        "desc": "Dixie Fire, Plumas County, CA — largest single wildfire in CA history",
    },
    "bootleg_fire": {
        "date": "2021-07-15",
        "lat": 42.49,
        "lon": -121.06,
        "desc": "Bootleg Fire, Fremont-Winema NF, OR — major 2021 Oregon fire",
    },
    "camp_fire": {
        "date": "2018-11-08",
        "lat": 39.81,
        "lon": -121.44,
        "desc": "Camp Fire, Butte County, CA — deadliest and most destructive CA wildfire",
    },
    "rim_fire": {
        "date": "2013-08-17",
        "lat": 37.87,
        "lon": -119.93,
        "desc": "Rim Fire, Stanislaus NF, CA — third-largest CA wildfire at time",
    },
}


def search_and_download(date: str, lat: float, lon: float, output_dir: str, radius_km: float = 200.0):
    """
    Search for MOD021KM granules that cover (lat, lon) on the given date
    and download them to output_dir.

    Returns list of downloaded file paths.
    """
    earthaccess.login(strategy="interactive", persist=True)

    # Bounding box: roughly radius_km around the center point
    deg = radius_km / 111.0
    bbox = (lon - deg, lat - deg, lon + deg, lat + deg)

    print(f"\nSearching MOD021KM for {date} within {radius_km} km of ({lat:.2f}, {lon:.2f})...")

    results = earthaccess.search_data(
        short_name="MOD021KM",
        temporal=(date, date),
        bounding_box=bbox,
    )

    if not results:
        print("No granules found. Try a wider radius or a different date.")
        return []

    print(f"Found {len(results)} granule(s):")
    for r in results:
        print(f"  {r['meta']['native-id']}")

    os.makedirs(output_dir, exist_ok=True)
    files = earthaccess.download(results, output_dir)
    print(f"\nDownloaded {len(files)} file(s) to '{output_dir}/'")
    return files


def main():
    parser = argparse.ArgumentParser(description="Download MODIS MOD021KM for wildfire events")
    parser.add_argument("--event", choices=list(FIRE_EVENTS.keys()), help="Named fire event")
    parser.add_argument("--list", action="store_true", help="List all built-in fire events")
    parser.add_argument("--lat", type=float, help="Center latitude (with --date and --lon)")
    parser.add_argument("--lon", type=float, help="Center longitude (with --date and --lat)")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format")
    parser.add_argument("--radius", type=float, default=200.0, help="Search radius in km (default 200)")
    parser.add_argument("--outdir", default="data", help="Output directory (default: data/)")
    args = parser.parse_args()

    if args.list:
        print("\nBuilt-in fire events:")
        for name, ev in FIRE_EVENTS.items():
            print(f"  {name:20s}  {ev['date']}  {ev['desc']}")
        return

    if args.lat and args.lon and args.date:
        search_and_download(args.date, args.lat, args.lon, args.outdir, args.radius)
    elif args.event:
        ev = FIRE_EVENTS[args.event]
        print(f"\nEvent: {ev['desc']}")
        search_and_download(ev["date"], ev["lat"], ev["lon"], args.outdir, args.radius)
    else:
        # Default: Creek Fire — a reliable, well-studied event
        ev = FIRE_EVENTS["creek_fire"]
        print(f"No event specified. Defaulting to: {ev['desc']}")
        print("Run with --list to see all available events, or --help for custom coordinates.")
        search_and_download(ev["date"], ev["lat"], ev["lon"], args.outdir, args.radius)


if __name__ == "__main__":
    main()
