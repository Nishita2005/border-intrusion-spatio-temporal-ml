import math
import numpy as np

def get_projection(lat, lon, speed, bearing, seconds):
    """Calculates a single future coordinate based on speed and bearing."""
    R = 6371000  # Earth's radius in meters
    distance = speed * seconds
    
    lat1, lon1 = math.radians(lat), math.radians(lon)
    brng = math.radians(bearing)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance/R) +
                     math.cos(lat1) * math.sin(distance/R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(distance/R) * math.cos(lat1),
                             math.cos(distance/R) - math.sin(lat1) * math.sin(lat2))
    
    return [math.degrees(lat2), math.degrees(lon2)]

def calculate_prediction_cone(lat, lon, speed, bearing, seconds=60):
    """Generates coordinates for a 30-degree uncertainty cone."""
    # Center Point (Most likely path)
    center = get_projection(lat, lon, speed, bearing, seconds)
    # Left Bound (+15 degrees)
    left = get_projection(lat, lon, speed, bearing - 15, seconds)
    # Right Bound (-15 degrees)
    right = get_projection(lat, lon, speed, bearing + 15, seconds)
    
    # Return as a list of points for a Folium Polygon [Current, Left, Right]
    return [[lat, lon], left, center, right, [lat, lon]]
def intersect(p1, q1, p2, q2):
    """Checks if line segment p1q1 and p2q2 intersect."""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    # Returns True if segments intersect
    return ccw(p1,p2,q2) != ccw(q1,p2,q2) and ccw(p1,q1,p2) != ccw(p1,q1,q2)

def check_breach(cone_coords, border_line):
    """
    Checks if the edges of the prediction cone cross the border line.
    cone_coords: [[lat, lon], left, center, right, [lat, lon]]
    border_line: [[lat, lon], [lat, lon], ...]
    """
    # Check the three main projection lines of the cone
    origin = cone_coords[0]
    projections = [cone_coords[1], cone_coords[2], cone_coords[3]]
    
    for proj in projections:
        for i in range(len(border_line) - 1):
            b1 = border_line[i]
            b2 = border_line[i+1]
            if intersect(origin, proj, b1, b2):
                return True
    return False