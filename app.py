import os
import sys
import time
import json
import threading
import uuid
import random
import logging
import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import sqlite3
from io import BytesIO
import base64
import math
import requests
from PIL import Image
from flask import Flask, render_template, request, Response, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ──────────────────────────────────────────────
# OPTIONAL DEPENDENCY IMPORTS WITH FALLBACKS
# ──────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Vehicle detection will be simulated.")

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

try:
    from shapely.geometry import Point, LineString
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
MODEL_PATH      = "yolov8n.pt"
UPLOAD_FOLDER   = "uploads"
RESULT_FOLDER   = "results"
LOG_FOLDER      = "logs"
STATIC_FOLDER   = "templates"

for folder in [UPLOAD_FOLDER, RESULT_FOLDER, LOG_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

IMG_SIZE     = 640
CONF_THRES   = 0.25
IOU_THRES    = 0.45
PX_PER_METER = 100.0

# Traffic-light timing
BASE_GREEN       = 8
MAX_GREEN        = 25
YELLOW_TIME      = 3
ALL_RED_TIME     = 1
SECONDS_PER_VEH  = 1.5

# COCO vehicle classes
VEHICLE_CLASS_MAP = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 1: 'bicycle'}

# Intelligence engine weights
VEHICLE_WEIGHTS = {'bicycles': 0.5, 'motorcycles': 0.5, 'cars': 1.0, 'buses': 3.0, 'trucks': 3.0}
DENSITY_THRESHOLDS = {'CRITICAL': 12, 'HIGH': 8, 'MEDIUM': 4}

# ──────────────────────────────────────────────
# GLOBAL CAMERA STATE
# ──────────────────────────────────────────────
camera_state = {
    'cap':    None,
    'active': False,
    'lock':   threading.Lock(),
    'thread': None,
}
detection_history = []

# ──────────────────────────────────────────────
# FLASK / SOCKETIO INIT  (eventlet for Render)
# ──────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=STATIC_FOLDER)
app.config['SECRET_KEY']           = os.getenv('FLASK_SECRET', 'dev_secret_key')
app.config['MAX_CONTENT_LENGTH']   = 100 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',          # eventlet works on Render
    max_http_buffer_size=100 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CUSTOM JSON ENCODER
# ──────────────────────────────────────────────
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):       return obj.isoformat()
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, np.integer):     return int(obj)
        if isinstance(obj, np.floating):    return float(obj)
        return super().default(obj)

app.json_encoder = CustomJSONEncoder


# ══════════════════════════════════════════════
# GEOCODING SERVICE
# ══════════════════════════════════════════════
class GeocodingService:
    FALLBACK = {
        "gandhipuram":  {"lat": 11.0168, "lng": 76.9558, "display": "Gandhipuram, Coimbatore"},
        "rs puram":     {"lat": 11.0180, "lng": 76.9700, "display": "RS Puram, Coimbatore"},
        "r s puram":    {"lat": 11.0180, "lng": 76.9700, "display": "RS Puram, Coimbatore"},
        "town hall":    {"lat": 11.0045, "lng": 76.9610, "display": "Town Hall, Coimbatore"},
        "peelamedu":    {"lat": 11.0275, "lng": 76.9425, "display": "Peelamedu, Coimbatore"},
        "singanallur":  {"lat": 11.0036, "lng": 77.0347, "display": "Singanallur, Coimbatore"},
        "race course":  {"lat": 11.0146, "lng": 76.9798, "display": "Race Course, Coimbatore"},
        "ukkadam":      {"lat": 10.9933, "lng": 76.9608, "display": "Ukkadam, Coimbatore"},
        "saibaba colony": {"lat": 11.0200, "lng": 76.9633, "display": "Saibaba Colony, Coimbatore"},
        "pappanaickenpalayam": {"lat": 11.0300, "lng": 76.9500, "display": "Pappanaickenpalayam, Coimbatore"},
        "hopes college": {"lat": 11.0215, "lng": 76.9570, "display": "Hopes College, Coimbatore"},
        "codissia":     {"lat": 11.0258, "lng": 76.9610, "display": "CODISSIA, Coimbatore"},
        "coimbatore junction": {"lat": 11.0017, "lng": 76.9674, "display": "Coimbatore Junction"},
        "railway station": {"lat": 11.0017, "lng": 76.9674, "display": "Coimbatore Railway Station"},
        "airport":      {"lat": 11.0300, "lng": 77.0434, "display": "Coimbatore Airport"},
        "tidel park":   {"lat": 11.0258, "lng": 76.9458, "display": "Tidel Park, Coimbatore"},
        "fun republic": {"lat": 11.0233, "lng": 76.9633, "display": "Fun Republic Mall, Coimbatore"},
        "brookefields": {"lat": 11.0175, "lng": 76.9833, "display": "Brookefields Mall, Coimbatore"},
    }

    def __init__(self):
        self.cache = {}
        if GEOPY_AVAILABLE:
            try:
                self.geolocator = Nominatim(user_agent="smart_traffic_cbr_v2", timeout=10)
                self.available  = True
            except Exception as e:
                logger.warning(f"Geocoding init failed: {e}")
                self.available = False
        else:
            self.available = False

    def _fallback_lookup(self, name: str):
        key = name.lower().strip()
        for fb_key, coords in self.FALLBACK.items():
            if fb_key in key or key in fb_key:
                return {'lat': coords['lat'], 'lng': coords['lng'],
                        'display_name': coords['display'], 'success': True}
        return None

    def geocode_location(self, location_name: str) -> dict:
        if location_name in self.cache:
            return self.cache[location_name]
        fb = self._fallback_lookup(location_name)
        if fb:
            self.cache[location_name] = fb
            return fb
        if self.available:
            try:
                loc = self.geolocator.geocode(f"{location_name}, Coimbatore, Tamil Nadu, India")
                if loc:
                    result = {'lat': loc.latitude, 'lng': loc.longitude,
                              'display_name': loc.address, 'success': True}
                    self.cache[location_name] = result
                    return result
            except Exception as e:
                logger.error(f"Geocoding error: {e}")
        return {'success': False, 'error': f'Location "{location_name}" not found'}

    def search_locations(self, query: str) -> dict:
        results = []
        q = query.lower().strip()
        for name, coords in self.FALLBACK.items():
            if q in name:
                results.append({'name': coords['display'], 'lat': coords['lat'], 'lng': coords['lng']})
        if self.available and len(results) < 3:
            try:
                locs = self.geolocator.geocode(
                    f"{query}, Coimbatore, Tamil Nadu, India",
                    exactly_one=False, limit=8
                )
                if locs:
                    for l in locs:
                        results.append({'name': l.address, 'lat': l.latitude, 'lng': l.longitude})
            except Exception as e:
                logger.error(f"Search error: {e}")
        return {'success': True, 'results': results[:10]}


# ══════════════════════════════════════════════
# ROAD NETWORK
# ══════════════════════════════════════════════
class CoimbatoreRoadNetwork:
    OSRM_BASE = "https://router.project-osrm.org/route/v1/driving"

    def __init__(self):
        self.graph        = None
        self.congestion   = {}
        self.intersections = []
        self._init_graph()

    def _init_graph(self):
        if NETWORKX_AVAILABLE and OSMNX_AVAILABLE:
            try:
                logger.info("Downloading Coimbatore road graph from OSM …")
                self.graph = ox.graph_from_point(
                    (11.0168, 76.9558), dist=8000,
                    network_type='drive', simplify=True
                )
                self._seed_congestion()
                self._identify_intersections()
                logger.info(f"OSM graph loaded – nodes={len(self.graph.nodes())}, "
                            f"edges={len(self.graph.edges())}")
                return
            except Exception as e:
                logger.warning(f"OSM graph failed ({e}), using fallback")
        self._init_fallback()

    def _seed_congestion(self):
        for u, v, *_ in self.graph.edges(keys=True):
            self.congestion[(u, v)] = random.uniform(0.0, 0.3)
            self.congestion[(v, u)] = self.congestion[(u, v)]

    def _identify_intersections(self):
        for nid, data in self.graph.nodes(data=True):
            if self.graph.degree(nid) >= 3:
                self.intersections.append({
                    'id': nid, 'lat': data['y'], 'lng': data['x'],
                    'name': f"Intersection {nid}", 'degree': self.graph.degree(nid)
                })
        self.intersections.sort(key=lambda x: x['degree'], reverse=True)

    def _init_fallback(self):
        LOCS = [
            (11.0168, 76.9558, "Gandhipuram"),
            (11.0180, 76.9700, "RS Puram"),
            (11.0045, 76.9610, "Town Hall"),
            (11.0275, 76.9425, "Peelamedu"),
            (11.0036, 77.0347, "Singanallur"),
            (11.0146, 76.9798, "Race Course"),
        ]
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
            for i, (lat, lng, name) in enumerate(LOCS):
                self.graph.add_node(i, y=lat, x=lng, name=name)
                self.intersections.append({'id': i, 'lat': lat, 'lng': lng, 'name': name, 'degree': 2})
            for i in range(len(LOCS)):
                for j in range(len(LOCS)):
                    if i != j:
                        d = self._haversine(LOCS[i][0], LOCS[i][1], LOCS[j][0], LOCS[j][1])
                        self.graph.add_edge(i, j, length=d * 1000, speed_limit=50, capacity=1000)
                        self.congestion[(i, j)] = 0.3
        else:
            self.graph = {'nodes': {}, 'edges': {}}
            for i, (lat, lng, name) in enumerate(LOCS):
                self.graph['nodes'][i] = {'y': lat, 'x': lng, 'name': name}
                self.intersections.append({'id': i, 'lat': lat, 'lng': lng, 'name': name, 'degree': 2})

    @staticmethod
    def _haversine(lat1, lng1, lat2, lng2) -> float:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lng2 - lng1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def get_route_with_real_roads(self, slat, slng, elat, elng, vehicle_type="car"):
        try:
            url = (f"{self.OSRM_BASE}/{slng},{slat};{elng},{elat}"
                   f"?overview=full&geometries=geojson&steps=false")
            resp = requests.get(url, timeout=8)
            data = resp.json()
            if data.get('code') == 'Ok' and data.get('routes'):
                route_data  = data['routes'][0]
                distance_km = route_data['distance'] / 1000
                duration_m  = route_data['duration'] / 60
                speed_factor = {'truck': 0.7, 'motorcycle': 1.1, 'bus': 0.75}.get(vehicle_type, 1.0)
                duration_m  /= speed_factor
                coords_raw = route_data['geometry']['coordinates']
                route_coords = [[c[1], c[0]] for c in coords_raw]
                return {
                    'success': True,
                    'route':           route_coords,
                    'distance':        round(distance_km, 2),
                    'estimated_time':  round(duration_m, 1),
                    'start':           [slat, slng],
                    'end':             [elat, elng],
                    'source':          'osrm',
                }
        except Exception as e:
            logger.warning(f"OSRM request failed: {e}. Trying local graph.")

        if NETWORKX_AVAILABLE and hasattr(self.graph, 'edges'):
            try:
                snode = self._nearest_node(slat, slng)
                enode = self._nearest_node(elat, elng)
                if snode is not None and enode is not None:
                    self._update_edge_weights(vehicle_type)
                    path = nx.shortest_path(self.graph, source=snode, target=enode, weight='travel_time')
                    coords, dist_km, dur_m = [], 0.0, 0.0
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edata = min(
                            (self.graph.get_edge_data(u, v) or {}).values(),
                            key=lambda x: x.get('length', float('inf')),
                            default={}
                        )
                        dist_km += edata.get('length', 0) / 1000
                        dur_m   += edata.get('travel_time', 0)
                        if 'geometry' in edata:
                            for lng, lat in edata['geometry'].coords:
                                coords.append([lat, lng])
                        else:
                            n = self.graph.nodes[u]
                            coords.append([n['y'], n['x']])
                    fn = self.graph.nodes[path[-1]]
                    coords.append([fn['y'], fn['x']])
                    return {
                        'success': True, 'route': coords,
                        'distance': round(dist_km, 2), 'estimated_time': round(dur_m, 1),
                        'start': [slat, slng], 'end': [elat, elng], 'source': 'osmnx',
                    }
            except Exception as e:
                logger.warning(f"Local graph routing failed: {e}")

        dist_km = self._haversine(slat, slng, elat, elng)
        dur_m   = (dist_km / 50) * 60
        return {
            'success': True,
            'route': [[slat, slng], [elat, elng]],
            'distance': round(dist_km, 2), 'estimated_time': round(dur_m, 1),
            'start': [slat, slng], 'end': [elat, elng], 'source': 'straight_line',
        }

    def _nearest_node(self, lat, lng):
        if OSMNX_AVAILABLE and hasattr(self.graph, 'nodes'):
            try:
                return ox.distance.nearest_nodes(self.graph, lng, lat)
            except Exception:
                pass
        min_d, nearest = float('inf'), None
        nodes = self.graph.nodes(data=True) if hasattr(self.graph, 'nodes') else []
        for nid, data in nodes:
            d = (lat - data.get('y', 0)) ** 2 + (lng - data.get('x', 0)) ** 2
            if d < min_d:
                min_d, nearest = d, nid
        return nearest

    def _update_edge_weights(self, vehicle_type):
        spd_pen = {'truck': 1.5, 'motorcycle': 0.8, 'bus': 1.3}.get(vehicle_type, 1.0)
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            length     = data.get('length', 100)
            speed      = data.get('speed_limit', 50)
            base_time  = (length / 1000) / speed * 60
            cong_pen   = 1.0 + self.congestion.get((u, v), 0.0) * 3.0
            data['travel_time'] = base_time * cong_pen * spd_pen

    def update_congestion(self, road, level):
        self.congestion[road] = min(1.0, max(0.0, level))
        try:
            socketio.emit("congestion_update", {'road': str(road), 'level': self.congestion[road]})
        except Exception:
            pass


# ══════════════════════════════════════════════
# TRAFFIC SIGNAL CONTROLLER
# ══════════════════════════════════════════════
class TrafficSignalController:
    def __init__(self, road_network):
        self.lanes           = ['NORTH', 'SOUTH', 'EAST', 'WEST']
        self.current_green   = 'NORTH'
        self.time_remaining  = BASE_GREEN
        self.phase           = 'GREEN'
        self.lane_densities  = {
            lane: {'total': 0, 'cars': 0, 'motorcycles': 0, 'buses': 0, 'trucks': 0, 'bicycles': 0}
            for lane in self.lanes
        }
        self.emergency_override = False
        self.emergency_lane     = None
        self.last_update        = time.time()
        self.cycle_history      = []
        self.road_network       = road_network

    def update_lane_density(self, lane, vehicle_counts):
        self.lane_densities[lane] = dict(vehicle_counts)
        self.lane_densities[lane]['total'] = sum(vehicle_counts.values())
        try:
            socketio.emit("lane_density_update", {'lane': lane, 'counts': vehicle_counts, 'timestamp': time.time()})
        except Exception:
            pass

    def calculate_optimal_green_time(self, lane):
        d = self.lane_densities.get(lane, {})
        w = (d.get('bicycles', 0)    * 0.3 +
             d.get('motorcycles', 0) * 0.5 +
             d.get('cars', 0)        * 1.0 +
             d.get('buses', 0)       * 2.0 +
             d.get('trucks', 0)      * 2.5)
        total = sum(l['total'] for l in self.lane_densities.values())
        max_g = MAX_GREEN * (1.5 if total > 50 else 1.0)
        return int(max(BASE_GREEN, min(BASE_GREEN + w * SECONDS_PER_VEH, max_g)))

    def get_next_lane(self):
        if self.emergency_override and self.emergency_lane:
            return self.emergency_lane, 100
        hour = datetime.now().hour
        peak = 1.5 if (8 <= hour <= 10 or 17 <= hour <= 20) else 1.0
        scores = {}
        for lane in self.lanes:
            if lane == self.current_green:
                continue
            d = self.lane_densities.get(lane, {})
            scores[lane] = (d.get('bicycles', 0) * 0.3 +
                            d.get('motorcycles', 0) * 0.5 +
                            d.get('cars', 0)        * 1.0 +
                            d.get('buses', 0)       * 1.5 +
                            d.get('trucks', 0)      * 2.0) * 2 * peak
        if scores:
            best = max(scores, key=scores.get)
            return best, scores[best]
        idx = (self.lanes.index(self.current_green) + 1) % len(self.lanes)
        return self.lanes[idx], 0

    def update_signal_state(self):
        now = time.time()
        if now - self.last_update < 1:
            return
        self.last_update = now
        self.time_remaining -= 1
        if self.time_remaining <= 0:
            if self.phase == 'GREEN':
                self.phase = 'YELLOW'; self.time_remaining = YELLOW_TIME
            elif self.phase == 'YELLOW':
                self.phase = 'RED'; self.time_remaining = ALL_RED_TIME
            elif self.phase == 'RED':
                nxt, pri = self.get_next_lane()
                self.current_green  = nxt
                self.phase          = 'GREEN'
                self.time_remaining = self.calculate_optimal_green_time(nxt)
                self.cycle_history.append({'lane': nxt, 'duration': self.time_remaining,
                                           'priority': pri, 'timestamp': now})
        try:
            socketio.emit("signal_update", {
                'current_lane':   self.current_green,
                'phase':          self.phase,
                'time_remaining': int(self.time_remaining),
                'emergency_mode': self.emergency_override,
                'emergency_lane': self.emergency_lane,
                'lane_densities': self.lane_densities,
            })
        except Exception:
            pass


# ══════════════════════════════════════════════
# VEHICLE DETECTOR
# ══════════════════════════════════════════════
class VehicleDetector:
    def __init__(self, model_path):
        self.model     = None
        self.available = False
        if not YOLO_AVAILABLE:
            return
        try:
            self.model     = YOLO(model_path if os.path.exists(model_path) else 'yolov8n.pt')
            self.available = True
            logger.info("YOLOv8 model loaded")
        except Exception as e:
            logger.error(f"YOLO load failed: {e}")

    def _simulate(self, image):
        h, w = image.shape[:2]
        vt   = ['cars', 'motorcycles', 'bicycles', 'buses', 'trucks']
        n    = random.randint(5, 15)
        dets = {k: 0 for k in vt}; dets['total'] = 0
        lanes = ['NORTH', 'SOUTH', 'EAST', 'WEST']
        lc    = {ln: {k: 0 for k in vt} for ln in lanes}
        bbs   = []
        for _ in range(n):
            vtype = random.choice(vt)
            lane  = random.choice(lanes)
            dets[vtype] += 1; dets['total'] += 1
            lc[lane][vtype] += 1
            x1 = random.randint(0, max(1, w // 2))
            y1 = random.randint(0, max(1, h // 2))
            x2 = min(x1 + random.randint(50, 150), w)
            y2 = min(y1 + random.randint(50, 150), h)
            bbs.append({'bbox': [x1, y1, x2, y2], 'class': vtype[:-1],
                        'type': vtype, 'confidence': round(random.uniform(0.7, 0.95), 2), 'lane': lane})
        return {'detections': dets, 'bounding_boxes': bbs, 'lane_counts': lc, 'total': n}

    def detect_vehicles(self, image):
        if not self.available:
            return self._simulate(image)
        try:
            results = self.model(image, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
            vt   = ['cars', 'motorcycles', 'bicycles', 'buses', 'trucks']
            dets = {k: 0 for k in vt}; dets['total'] = 0
            lanes = ['NORTH', 'SOUTH', 'EAST', 'WEST']
            lc    = {ln: {k: 0 for k in vt} for ln in lanes}
            bbs   = []
            TYPE_MAP = {'car': 'cars', 'motorcycle': 'motorcycles', 'bicycle': 'bicycles',
                        'bus': 'buses', 'truck': 'trucks'}
            h, w = image.shape[:2]
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls  = int(box.cls[0])
                    vt_s = VEHICLE_CLASS_MAP.get(cls)
                    if not vt_s:
                        continue
                    key  = TYPE_MAP[vt_s]
                    conf = float(box.conf[0])
                    dets[key] += 1; dets['total'] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    lane = ('NORTH' if cy < h / 2 else 'SOUTH') if cx < w / 2 else \
                           ('EAST'  if cy < h / 2 else 'WEST')
                    lc[lane][key] += 1
                    bbs.append({'bbox': [x1, y1, x2, y2], 'class': vt_s, 'type': key,
                                'confidence': round(conf, 2), 'lane': lane})
            return {'detections': dets, 'bounding_boxes': bbs, 'lane_counts': lc, 'total': dets['total']}
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return self._simulate(image)

    def draw_detections(self, image, bounding_boxes, traffic_analysis=None):
        colors = {'cars': (0,255,0), 'motorcycles': (255,255,0),
                  'bicycles': (0,255,255), 'buses': (255,0,255), 'trucks': (255,0,0)}
        level_colors = {'CRITICAL': (0,0,255), 'HIGH': (0,100,255),
                        'MEDIUM': (0,165,255), 'LOW': (0,200,0)}
        for bb in bounding_boxes:
            x1,y1,x2,y2 = bb['bbox']
            color = colors.get(bb['type'], (255,255,255))
            cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
            lbl = f"{bb['type'][:-1]} [{bb.get('lane','')}] {int(bb['confidence']*100)}%"
            cv2.putText(image, lbl, (x1, max(y1-10,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if traffic_analysis:
            s  = traffic_analysis.get('summary', traffic_analysis)
            lr = s.get('lane_ranking', [])
            pri= s.get('priority_lane', '')
            gd = s.get('green_duration', 0)
            dl = s.get('density_level', '')
            ds = s.get('density_score', 0)
            h, w = image.shape[:2]
            pw, lh = 240, 24
            ph = 32 + len(lr) * lh
            ov = image.copy()
            cv2.rectangle(ov, (10,10), (10+pw, 10+ph), (20,20,20), -1)
            cv2.addWeighted(ov, 0.6, image, 0.4, 0, image)
            cv2.putText(image, "LANE DENSITY ANALYSIS", (15,28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,242,255), 1)
            for i, item in enumerate(lr):
                iy = 30 + (i+1)*lh
                lc2 = level_colors.get(item['level'], (180,180,180))
                pfx = ">> " if item.get('is_priority') else "   "
                cv2.putText(image, f"{pfx}{item['lane']}: {item['score']:.1f}pts [{item['level']}]",
                            (15, iy+16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, lc2, 1)
            bh = 44; by = h - bh
            bov = image.copy()
            cv2.rectangle(bov, (0,by), (w,h), (15,15,15), -1)
            cv2.addWeighted(bov, 0.65, image, 0.35, 0, image)
            bc = level_colors.get(dl, (60,60,60))
            cv2.putText(image,
                        f"AI SIGNAL: GREEN->{pri}  |  {gd}s  |  {dl} ({ds:.1f}pts)",
                        (10, by+28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2)
        return image


# ══════════════════════════════════════════════
# TRAFFIC INTELLIGENCE ENGINE
# ══════════════════════════════════════════════
class TrafficIntelligence:
    @staticmethod
    def extract_features(lane_counts: dict) -> dict:
        features = {}
        for lane, counts in lane_counts.items():
            features[lane] = {
                'bicycles':    counts.get('bicycles', 0),
                'motorcycles': counts.get('motorcycles', 0),
                'cars':        counts.get('cars', 0),
                'buses':       counts.get('buses', 0),
                'trucks':      counts.get('trucks', 0),
                'total':       sum(counts.values()),
            }
        return features

    @staticmethod
    def compute_density_scores(features: dict) -> dict:
        density = {}
        for lane, c in features.items():
            score = round(
                c.get('bicycles', 0)    * 0.5 +
                c.get('motorcycles', 0) * 0.5 +
                c.get('cars', 0)        * 1.0 +
                c.get('buses', 0)       * 3.0 +
                c.get('trucks', 0)      * 3.0, 2)
            level = ('CRITICAL' if score >= 12 else 'HIGH' if score >= 8
                     else 'MEDIUM' if score >= 4 else 'LOW')
            density[lane] = {'score': score, 'level': level}
        return density

    @staticmethod
    def predict_signal(density_scores: dict) -> dict:
        if not density_scores:
            return {'priority_lane': 'NORTH', 'ranked_lanes': [],
                    'reason': 'No vehicles detected. Default signal applied.'}
        ranked = sorted(density_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        pri = ranked[0][0]; top = ranked[0][1]
        return {
            'priority_lane': pri,
            'ranked_lanes': [{'lane': l, 'score': d['score'], 'level': d['level']} for l, d in ranked],
            'reason': f"Lane {pri} has highest occupancy ({top['score']} pts, {top['level']}). AI→GREEN:{pri}.",
        }

    @staticmethod
    def predict_green_duration(priority_lane: str, density_scores: dict) -> dict:
        if priority_lane not in density_scores:
            return {'duration_sec': 20, 'level': 'LOW', 'score': 0}
        score = density_scores[priority_lane]['score']
        level = density_scores[priority_lane]['level']
        dur   = 60 if score >= 6 else 40 if score >= 3 else 20
        return {'duration_sec': dur, 'level': level, 'score': score}

    @classmethod
    def analyse(cls, lane_counts: dict) -> dict:
        f1 = cls.extract_features(lane_counts)
        f2 = cls.compute_density_scores(f1)
        f3 = cls.predict_signal(f2)
        f4 = cls.predict_green_duration(f3['priority_lane'], f2)
        lane_display = [{
            'lane': item['lane'], 'score': item['score'], 'level': item['level'],
            'total':       f1.get(item['lane'], {}).get('total', 0),
            'cars':        f1.get(item['lane'], {}).get('cars', 0),
            'motorcycles': f1.get(item['lane'], {}).get('motorcycles', 0),
            'bicycles':    f1.get(item['lane'], {}).get('bicycles', 0),
            'buses':       f1.get(item['lane'], {}).get('buses', 0),
            'trucks':      f1.get(item['lane'], {}).get('trucks', 0),
            'is_priority': item['lane'] == f3['priority_lane'],
        } for item in f3['ranked_lanes']]
        return {
            'step1_features': f1, 'step2_density': f2,
            'step3_signal': f3,   'step4_green_time': f4,
            'summary': {
                'priority_lane':  f3['priority_lane'],
                'green_duration': f4['duration_sec'],
                'density_level':  f4['level'],
                'density_score':  f4['score'],
                'reason':         f3['reason'],
                'lane_ranking':   lane_display,
            },
        }


# ══════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════
def init_database():
    try:
        conn = sqlite3.connect('traffic_analytics.db', check_same_thread=False)
        cur  = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            detection_type TEXT, file_name TEXT, total_vehicles INTEGER,
            cars INTEGER, motorcycles INTEGER, bicycles INTEGER, buses INTEGER, trucks INTEGER,
            processing_time REAL, lane_data TEXT)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS signal_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            lane TEXT, phase TEXT, duration INTEGER, vehicle_count INTEGER, optimization_score REAL)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS route_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            start_location TEXT, end_location TEXT, distance REAL, estimated_time REAL, vehicle_type TEXT)''')
        conn.commit(); conn.close()
        logger.info("Database initialised")
        return True
    except Exception as e:
        logger.error(f"DB init error: {e}")
        return False


# ══════════════════════════════════════════════
# GLOBAL INSTANCES
# ══════════════════════════════════════════════
geocoding_service  = GeocodingService()
road_network       = CoimbatoreRoadNetwork()
signal_controller  = TrafficSignalController(road_network)
vehicle_detector   = VehicleDetector(MODEL_PATH)

current_stats = {
    'vehicle_count': 0,
    'total_detections': {'cars': 0, 'motorcycles': 0, 'bicycles': 0, 'buses': 0, 'trucks': 0},
    'avg_speed': 45.2, 'violations': 0, 'accuracy': 94.7,
    'lane_densities': signal_controller.lane_densities,
    'emergency_events': 0, 'last_updated': datetime.now().isoformat()
}
stats_lock = threading.Lock()


# ══════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════
@app.route('/')
def index():
    try:    return render_template('index.html')
    except Exception as e: return f"Error: {e}", 500

@app.route('/input')
def input_page():
    try:    return render_template('input.html')
    except Exception as e: return f"Error: {e}", 500

@app.route('/dashboard')
def dashboard_page():
    try:    return render_template('dashboard.html')
    except Exception as e: return f"Error: {e}", 500

@app.route('/map')
def map_page():
    try:    return render_template('map.html')
    except Exception as e: return f"Error: {e}", 500

@app.route('/analytics')
def analytics_page():
    try:    return render_template('analytics.html')
    except Exception as e: return f"Error: {e}", 500

@app.route('/api/stats')
def api_stats():
    with stats_lock:
        return jsonify(current_stats)

@app.route('/api/signal/status')
def api_signal_status():
    return jsonify({
        'current_lane':   signal_controller.current_green,
        'phase':          signal_controller.phase,
        'time_remaining': signal_controller.time_remaining,
        'lane_densities': signal_controller.lane_densities,
        'emergency_mode': signal_controller.emergency_override,
    })

@app.route('/api/geocode/search', methods=['POST'])
def api_geocode_search():
    data  = request.get_json() or {}
    query = data.get('query', '')
    if len(query) < 2:
        return jsonify({'success': True, 'results': []})
    return jsonify(geocoding_service.search_locations(query))

@app.route('/api/geocode/location', methods=['POST'])
def api_geocode_location():
    data  = request.get_json() or {}
    name  = data.get('location', '')
    if not name:
        return jsonify({'success': False, 'error': 'Location required'}), 400
    return jsonify(geocoding_service.geocode_location(name))

@app.route('/api/route/calculate', methods=['POST'])
def api_calculate_route():
    try:
        data         = request.get_json() or {}
        start        = data.get('start')
        end          = data.get('end')
        vehicle_type = data.get('vehicle_type', 'car')
        if not start or not end:
            return jsonify({'success': False, 'error': 'start and end required'}), 400

        def resolve(loc):
            if isinstance(loc, dict):
                return loc.get('lat'), loc.get('lng')
            r = geocoding_service.geocode_location(str(loc))
            if r.get('success'):
                return r['lat'], r['lng']
            return None, None

        slat, slng = resolve(start)
        elat, elng = resolve(end)
        if None in (slat, slng, elat, elng):
            return jsonify({'success': False, 'error': 'Could not geocode one or both locations'}), 400

        result = road_network.get_route_with_real_roads(slat, slng, elat, elng, vehicle_type)
        try:
            conn = sqlite3.connect('traffic_analytics.db')
            conn.execute('INSERT INTO route_logs (start_location,end_location,distance,estimated_time,vehicle_type) VALUES (?,?,?,?,?)',
                         (str(start), str(end), result.get('distance', 0),
                          result.get('estimated_time', 0), vehicle_type))
            conn.commit(); conn.close()
        except Exception:
            pass
        return jsonify(result)
    except Exception as e:
        logger.error(f"Route API error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect/image', methods=['POST'])
def api_detect_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        f   = request.files['image']
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image'}), 400

        t0      = time.time()
        results = vehicle_detector.detect_vehicles(img)
        proc_t  = time.time() - t0
        analysis = TrafficIntelligence.analyse(results['lane_counts'])
        summary  = analysis['summary']

        if summary['priority_lane'] in signal_controller.lanes:
            signal_controller.current_green  = summary['priority_lane']
            signal_controller.time_remaining = summary['green_duration']

        annotated = vehicle_detector.draw_detections(img.copy(), results['bounding_boxes'], analysis)
        cv2.imwrite(os.path.join(RESULT_FOLDER, f"det_{int(t0)}.jpg"), annotated)
        _, buf = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buf).decode()

        for lane, counts in results['lane_counts'].items():
            signal_controller.update_lane_density(lane, counts)

        try: socketio.emit('traffic_intelligence', summary)
        except Exception: pass

        with stats_lock:
            current_stats['vehicle_count'] += results['total']
            for k in results['detections']:
                if k != 'total':
                    current_stats['total_detections'][k] += results['detections'][k]
            current_stats['last_updated'] = datetime.now().isoformat()

        try:
            conn = sqlite3.connect('traffic_analytics.db')
            conn.execute('INSERT INTO detection_logs (detection_type,file_name,total_vehicles,cars,motorcycles,bicycles,buses,trucks,processing_time,lane_data) VALUES (?,?,?,?,?,?,?,?,?,?)',
                         ('image', f.filename, results['total'],
                          results['detections']['cars'], results['detections']['motorcycles'],
                          results['detections']['bicycles'], results['detections']['buses'],
                          results['detections']['trucks'], proc_t, json.dumps(results['lane_counts'])))
            conn.commit(); conn.close()
        except Exception: pass

        return jsonify({'success': True, 'results': results, 'processing_time': round(proc_t, 3),
                        'annotated_image': img_b64, 'traffic_analysis': analysis})
    except Exception as e:
        logger.error(f"Image detect error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect/video', methods=['POST'])
def api_detect_video():
    filepath = None
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video provided'}), 400
        f    = request.files['video']
        ext  = os.path.splitext(f.filename)[1] or '.mp4'
        filepath = os.path.join(UPLOAD_FOLDER, f"tmp_{int(time.time())}{ext}")
        f.save(filepath)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Cannot open video'}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vt_keys = ['cars', 'motorcycles', 'bicycles', 'buses', 'trucks']
        total_dets = {k: 0 for k in vt_keys}; total_dets['total'] = 0
        lanes_acc  = {ln: {k: 0 for k in vt_keys} for ln in ['NORTH','SOUTH','EAST','WEST']}
        fc = 0; pf = 0; t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break
            if fc % 10 == 0:
                r = vehicle_detector.detect_vehicles(frame)
                for k in vt_keys: total_dets[k] += r['detections'].get(k, 0)
                total_dets['total'] += r['total']
                for lane, counts in r['lane_counts'].items():
                    for k, v in counts.items(): lanes_acc[lane][k] += v
                pf += 1
                if pf % 10 == 0:
                    try:
                        socketio.emit('video_progress', {
                            'frame': fc, 'total_frames': total_frames,
                            'processed': pf, 'progress': min(fc/max(total_frames,1)*100, 100),
                            'detections': total_dets})
                    except Exception: pass
            fc += 1
            if fc > 10000: break

        cap.release()
        proc_t = time.time() - t0
        try: os.remove(filepath)
        except Exception: pass

        if pf > 0:
            for lane in lanes_acc:
                for k in lanes_acc[lane]: lanes_acc[lane][k] //= pf

        for lane, counts in lanes_acc.items():
            signal_controller.update_lane_density(lane, counts)

        analysis = TrafficIntelligence.analyse(lanes_acc)
        summary  = analysis['summary']
        if summary['priority_lane'] in signal_controller.lanes:
            signal_controller.current_green  = summary['priority_lane']
            signal_controller.time_remaining = summary['green_duration']
        try: socketio.emit('traffic_intelligence', summary)
        except Exception: pass

        try: socketio.emit('video_progress', {'frame': fc, 'total_frames': fc, 'processed': pf,
                                              'progress': 100, 'detections': total_dets, 'complete': True})
        except Exception: pass

        with stats_lock:
            current_stats['vehicle_count'] += total_dets['total']
            for k in vt_keys: current_stats['total_detections'][k] += total_dets[k]
            current_stats['last_updated'] = datetime.now().isoformat()

        return jsonify({'success': True, 'results': total_dets, 'frames_total': fc,
                        'frames_processed': pf, 'processing_time': round(proc_t, 2),
                        'lane_data': lanes_acc, 'traffic_analysis': analysis})
    except Exception as e:
        logger.error(f"Video detect error: {e}", exc_info=True)
        if filepath and os.path.exists(filepath):
            try: os.remove(filepath)
            except Exception: pass
        return jsonify({'success': False, 'error': str(e)}), 500


# ══════════════════════════════════════════════
# LIVE CAMERA  (disabled on Render – no webcam)
# ══════════════════════════════════════════════
@app.route('/api/camera/start', methods=['POST', 'GET'])
def api_camera_start():
    # Render servers have no physical camera; return graceful message
    return jsonify({
        'success': False,
        'error': 'Live camera is not available on cloud deployments. '
                 'Please use image or video upload instead.'
    }), 400

@app.route('/api/camera/stop', methods=['POST', 'GET'])
def api_camera_stop():
    return jsonify({'success': True, 'message': 'Camera not active'})


# ══════════════════════════════════════════════
# SOCKETIO EVENTS
# ══════════════════════════════════════════════
@socketio.on('connect')
def handle_connect():
    try:
        emit("connected", {"message": "Connected to Smart Traffic System"})
        emit("stats_update", current_stats)
        emit("signal_update", {
            'current_lane':   signal_controller.current_green,
            'phase':          signal_controller.phase,
            'time_remaining': signal_controller.time_remaining,
            'lane_densities': signal_controller.lane_densities,
        })
        logger.info("Client connected")
    except Exception as e:
        logger.error(f"connect handler error: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")


# ══════════════════════════════════════════════
# BACKGROUND THREADS
# ══════════════════════════════════════════════
def signal_update_thread():
    logger.info("Signal thread started")
    while True:
        try:
            signal_controller.update_signal_state()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Signal thread error: {e}")
            time.sleep(5)

def stats_update_thread():
    logger.info("Stats thread started")
    while True:
        try:
            with stats_lock:
                current_stats['last_updated'] = datetime.now().isoformat()
                try: socketio.emit('stats_update', current_stats)
                except Exception: pass
            time.sleep(5)
        except Exception as e:
            logger.error(f"Stats thread error: {e}")
            time.sleep(10)


# ══════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):    return jsonify({'success': False, 'error': 'Not found'}), 404
@app.errorhandler(500)
def server_error(e): return jsonify({'success': False, 'error': 'Internal error'}), 500
@app.errorhandler(413)
def too_large(e):    return jsonify({'success': False, 'error': 'File too large (max 100 MB)'}), 413


# ══════════════════════════════════════════════
# MAIN  –  start background threads then serve
# ══════════════════════════════════════════════
init_database()

threading.Thread(target=signal_update_thread, daemon=True, name="SignalThread").start()
threading.Thread(target=stats_update_thread,  daemon=True, name="StatsThread").start()

if __name__ == "__main__":
    print("=" * 70)
    print("🚦  AI-Enabled Smart Traffic Management System — Coimbatore")
    print("=" * 70)
    print(f"   YOLO       : {'Loaded' if vehicle_detector.available else 'Simulated'}")
    print(f"   Road graph : {'OSM' if OSMNX_AVAILABLE else 'Fallback'}")
    print(f"   Routing    : OSRM → OSM graph → straight-line")
    print(f"   Geocoding  : {'Nominatim+Fallback' if geocoding_service.available else 'Fallback dict'}")
    print("=" * 70)

    port = int(os.getenv("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)