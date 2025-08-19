"""
geo_utils.py — Wildlife Location Metadata Utility
---------------------------------------------------

This module provides location utilities for the Wildlife Vision System, including:

- Reverse geocoding latitude/longitude to structured location metadata
- Geocoding place names using OpenStreetMap Nominatim
- Scraping Wikipedia infoboxes for national park or region names
- Normalizing state and country codes for consistent storage

Expected secrets:
- USER_AGENT defined in `.streamlit/secrets.toml`

Dependencies:
- requests for API calls
- BeautifulSoup for Wikipedia parsing
- OpenStreetMap Nominatim for geocoding

"""

import requests
from bs4 import BeautifulSoup
import json
import streamlit as st

# --- User-Agent for API Calls ---
USER_AGENT = st.secrets["USER_AGENT"]

# --- Mappings ---
us_state_to_abbrev = {
    # US states to 2-letter abbreviations
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
    'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
    'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC', 'puerto rico': 'PR',
    'guam': 'GU', 'virgin islands': 'VI', 'american samoa': 'AS', 'northern mariana islands': 'MP'
}

country_alpha2_to_alpha3 = {
    'US': 'USA', 'CA': 'CAN', 'MX': 'MEX',
    'GB': 'GBR', 'DE': 'DEU', 'FR': 'FRA', 'AU': 'AUS',
}


def scrape_wikipedia_infobox_location(wikipedia_slug: str) -> str | None:
    """
    Extracts location name from a Wikipedia infobox, if available.

    Args:
        wikipedia_slug (str): Wikipedia slug (e.g., 'Yellowstone_National_Park')

    Returns:
        str | None: Location name if found, else None
    """
    url = f"https://en.wikipedia.org/wiki/{wikipedia_slug}"
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        infobox = soup.find("table", {"class": "infobox"})
        if not infobox:
            return None

        for row in infobox.find_all("tr"):
            header = row.find("th")
            if header and "location" in header.text.lower():
                cell = row.find("td")
                if cell:
                    links = cell.find_all("a")
                    return links[0].get_text().strip() if links else cell.get_text(strip=True).split(',')[0]
    except Exception as e:
        print(f"Failed to scrape infobox location for {wikipedia_slug}: {e}")
    return None


def normalize_location_fields(address: dict, extra_tags: dict, name_details: dict, latitude: float, longitude: float) -> dict:
    """
    Converts raw geocoding results into normalized location metadata fields.

    Args:
        address (dict): Address fields from Nominatim
        extra_tags (dict): Extra tags from Nominatim
        name_details (dict): Name details from Nominatim
        latitude (float)
        longitude (float)

    Returns:
        dict: Normalized fields (country, state, park, etc.)
    """
    country_code_alpha2 = address.get('country_code', '').upper()
    country_code_alpha3 = country_alpha2_to_alpha3.get(country_code_alpha2, "Unknown")

    state_name = address.get('state', '')
    admin1_region_code = us_state_to_abbrev.get(state_name.lower(), "Unknown") if country_code_alpha2 == 'US' else "Unknown"

    specific_place_name = (
        address.get('locality') or name_details.get('name:en') or name_details.get('name') or "Unknown"
    )

    wikipedia_slug = extra_tags.get('wikidata') or extra_tags.get('wikipedia', '').split(':', 1)[-1]
    national_park_name = scrape_wikipedia_infobox_location(wikipedia_slug) if wikipedia_slug else "Unknown"

    county = address.get('county', "Unknown")

    parts = [specific_place_name, national_park_name, county, state_name, address.get('country')]
    display_parts = [part.strip() for part in parts if part and part.strip() and part.strip().lower() != specific_place_name.lower()]

    return {
        "latitude": latitude,
        "longitude": longitude,
        "country": country_code_alpha3,
        "state": admin1_region_code,
        "place": specific_place_name,
        "park": national_park_name,
        "county": county,
        "full_display_name": ", ".join([specific_place_name] + display_parts)
    }


def get_location_metadata(latitude: float, longitude: float) -> dict:
    """
    Reverse geocodes coordinates into structured location metadata.

    Args:
        latitude (float)
        longitude (float)

    Returns:
        dict: Normalized fields (country, state, place, park, etc.)
    """
    base_url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": latitude, "lon": longitude, "format": "json",
        "addressdetails": 1, "extratags": 1, "namedetails": 1, "accept-language": "en"
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return normalize_location_fields(
            address=data.get('address', {}),
            extra_tags=data.get('extratags', {}),
            name_details=data.get('namedetails', {}),
            latitude=float(data['lat']),
            longitude=float(data['lon'])
        )
    except Exception as e:
        print(f"Error during reverse geocoding: {e}")
        return {}


def geocode_place_name(place_name: str) -> dict:
    """
    Geocodes a place name into coordinates and structured location metadata.

    Args:
        place_name (str): e.g., "Yellowstone National Park"

    Returns:
        dict: Normalized location metadata, or defaults if not found
    """
    base_url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": USER_AGENT}
    params = {
        "q": place_name, "format": "json", "limit": 1,
        "addressdetails": 1, "extratags": 1, "namedetails": 1, "accept-language": "en"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            item = data[0]
            return normalize_location_fields(
                address=item.get('address', {}),
                extra_tags=item.get('extratags', {}),
                name_details=item.get('namedetails', {}),
                latitude=float(item['lat']),
                longitude=float(item['lon'])
            )
    except Exception as e:
        print(f"Error geocoding '{place_name}': {e}")

    return {
        "latitude": 0.0, "longitude": 0.0, "country": "Unknown", "state": "Unknown",
        "place": "Unknown", "park": "Unknown", "county": "Unknown",
        "full_display_name": "Unknown"
    }


if __name__ == "__main__":
    # Example Test Block
    lat = 44.66935
    lon = -110.4716

    print("\nLocation Metadata (Hayden Valley):")
    print(json.dumps(get_location_metadata(lat, lon), indent=2))

    print("\nGeocode Example (Yellowstone):")
    print(json.dumps(geocode_place_name("Yellowstone National Park"), indent=2))
