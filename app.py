
import math
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# -------------------- PAGE SETUP --------------------

st.set_page_config(
    page_title="Drone Delivery Route Optimizer",
    page_icon="🚁",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "route_config.json"
RESULTS_FILE = BASE_DIR / "experiment_results.json"

st.title("🚁 Drone Delivery Route Optimization")
st.caption("Genetic Algorithm • GPS distance • Interactive route map")

# -------------------- LOAD KAGGLE EXPERIMENT --------------------

config = {}
results = {}

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

if RESULTS_FILE.exists():
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

with st.expander("Kaggle experiment details", expanded=False):
    st.write("Algorithm:", config.get("algorithm", "Genetic Algorithm"))
    st.write("Dataset:", config.get("dataset_file", "Not available"))
    st.write(
        "Kaggle experiment distance:",
        results.get("optimized_distance_grid_units", "Not available"),
        "grid units"
    )
    st.info(
        "Kaggle coordinates are grid positions. The live GPS route below "
        "is a separate experiment using geographic coordinates."
    )

# -------------------- CITY PRESETS --------------------

cities = {
    "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Vijayawada": (16.5062, 80.6480),
    "Visakhapatnam": (17.6868, 83.2185),
    "Pune": (18.5204, 73.8567)
}

# -------------------- DISTANCE CALCULATION --------------------

def haversine(point_a, point_b):
    """Return great-circle distance in kilometres."""
    lat1, lon1 = point_a
    lat2, lon2 = point_b

    earth_radius_km = 6371.0

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius_km * math.asin(
        math.sqrt(min(1.0, a))
    )


def route_distance(route, points):
    """Distance from warehouse through deliveries and back."""
    sequence = [0] + [i + 1 for i in route] + [0]

    return sum(
        haversine(points[sequence[i]], points[sequence[i + 1]])
        for i in range(len(sequence) - 1)
    )


# -------------------- GENETIC ALGORITHM --------------------

def crossover(parent1, parent2):
    size = len(parent1)

    if size < 2:
        return parent1.copy()

    left, right = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[left:right + 1] = parent1[left:right + 1]

    remaining = [x for x in parent2 if x not in child]
    empty_positions = [i for i, x in enumerate(child) if x is None]

    for position, value in zip(empty_positions, remaining):
        child[position] = value

    return child


def genetic_algorithm(points, population_size=80, generations=150,
                      mutation_rate=0.15, seed=42):
    """Optimize delivery visit order; index 0 is the warehouse."""
    n = len(points) - 1

    if n == 0:
        return [], [0.0]

    random.seed(seed)
    np.random.seed(seed)

    baseline = list(range(n))

    def distance(route):
        return route_distance(route, points)

    population = [
        random.sample(range(n), n)
        for _ in range(population_size)
    ]

    population[0] = baseline.copy()

    best_route = baseline.copy()
    best_distance = distance(best_route)
    history = []

    for _ in range(generations):
        population.sort(key=distance)

        if distance(population[0]) < best_distance:
            best_route = population[0].copy()
            best_distance = distance(best_route)

        history.append(best_distance)

        next_population = [
            route.copy() for route in population[:2]
        ]

        while len(next_population) < population_size:
            candidates1 = random.sample(
                population, min(4, len(population))
            )
            candidates2 = random.sample(
                population, min(4, len(population))
            )

            parent1 = min(candidates1, key=distance)
            parent2 = min(candidates2, key=distance)

            child = crossover(parent1, parent2)

            if len(child) >= 2 and random.random() < mutation_rate:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]

            next_population.append(child)

        population = next_population

    candidate = min(population, key=distance)

    if distance(candidate) < best_distance:
        best_route = candidate.copy()
        best_distance = distance(best_route)

    history.append(best_distance)

    return best_route, history


# -------------------- INPUT FORM --------------------

st.header("1. Enter delivery locations")

selected_city = st.selectbox("Select city", list(cities.keys()))
center_lat, center_lon = cities[selected_city]

st.write(
    "City coordinates are used as an approximate map centre. "
    "You can enter your own warehouse and delivery coordinates below."
)

with st.form("route_form"):
    col1, col2 = st.columns(2)

    with col1:
        warehouse_lat = st.number_input(
            "Warehouse latitude",
            min_value=-90.0, max_value=90.0,
            value=float(center_lat), format="%.6f"
        )

        warehouse_lon = st.number_input(
            "Warehouse longitude",
            min_value=-180.0, max_value=180.0,
            value=float(center_lon), format="%.6f"
        )

    with col2:
        num_deliveries = st.slider(
            "Number of delivery locations", 2, 12, 5
        )

        population_size = st.select_slider(
            "GA population size",
            options=[30, 50, 80, 100, 150],
            value=80
        )

        generations = st.select_slider(
            "GA generations",
            options=[50, 100, 150, 200, 300],
            value=150
        )

    st.markdown("#### Delivery GPS coordinates")

    delivery_inputs = []

    for i in range(num_deliveries):
        st.markdown(f"**Delivery {i + 1}**")
        c1, c2 = st.columns(2)

        # Small offsets create editable example locations near the city centre.
        default_lat = max(
            -90.0, min(90.0, center_lat + 0.015 * (i + 1))
        )
        default_lon = max(
            -180.0, min(180.0, center_lon + 0.012 * (i + 1))
        )

        with c1:
            lat = st.number_input(
                f"Latitude {i + 1}",
                min_value=-90.0, max_value=90.0,
                value=float(default_lat),
                format="%.6f", key=f"lat_{i}"
            )

        with c2:
            lon = st.number_input(
                f"Longitude {i + 1}",
                min_value=-180.0, max_value=180.0,
                value=float(default_lon),
                format="%.6f", key=f"lon_{i}"
            )

        delivery_inputs.append((lat, lon))

    submitted = st.form_submit_button(
        "🚀 Optimize Delivery Route",
        type="primary"
    )


# -------------------- OPTIMIZATION RESULTS --------------------


# -------------------- OPTIMIZATION RESULTS --------------------

# Preserve results when Streamlit reruns the app.
if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None

# Run the algorithm only after form submission.
if submitted:
    points = [(warehouse_lat, warehouse_lon)] + delivery_inputs

    if len(set(points[1:])) != len(points[1:]):
        st.error("Two or more delivery locations have identical coordinates.")
        st.stop()

    with st.spinner("Searching for a shorter route..."):
        baseline_route = list(range(len(delivery_inputs)))

        baseline_distance = route_distance(baseline_route, points)

        best_route, history = genetic_algorithm(
            points,
            population_size=population_size,
            generations=generations,
            mutation_rate=0.15,
            seed=42
        )

        optimized_distance = route_distance(best_route, points)

    improvement = (
        (baseline_distance - optimized_distance) / baseline_distance * 100
        if baseline_distance > 0 else 0.0
    )

    # Save all results so they survive future reruns.
    st.session_state.optimization_result = {
        "city": selected_city,
        "points": points,
        "baseline_distance": baseline_distance,
        "optimized_distance": optimized_distance,
        "improvement": improvement,
        "best_route": best_route,
        "history": history,
        "population_size": population_size,
        "generations": generations
    }

# Display the saved results.
result = st.session_state.optimization_result

if result is None:
    st.info(
        "Choose a city, enter the warehouse and delivery coordinates, "
        "then click Optimize Delivery Route."
    )
else:
    points = result["points"]
    best_route = result["best_route"]
    history = result["history"]
    baseline_distance = result["baseline_distance"]
    optimized_distance = result["optimized_distance"]
    improvement = result["improvement"]

    st.header("2. Optimization results")

    m1, m2, m3 = st.columns(3)

    m1.metric("Original distance", f"{baseline_distance:.2f} km")
    m2.metric(
        "Optimized distance",
        f"{optimized_distance:.2f} km",
        delta=f"{optimized_distance - baseline_distance:.2f} km"
    )
    m3.metric("Distance reduction", f"{improvement:.2f}%")

    st.caption(
        "Distance is estimated using the Haversine formula. "
        "It represents straight-line surface distance, not actual "
        "road distance or a certified drone flight path."
    )

    st.header("3. Optimized visit sequence")

    visit_sequence = ["Warehouse"] + [
        f"Delivery {i + 1}" for i in best_route
    ] + ["Warehouse"]

    st.write(" → ".join(visit_sequence))

    route_table = pd.DataFrame([
        {
            "Visit order": order,
            "Location": f"Delivery {delivery_index + 1}",
            "Latitude": points[delivery_index + 1][0],
            "Longitude": points[delivery_index + 1][1]
        }
        for order, delivery_index in enumerate(best_route, start=1)
    ])

    st.dataframe(
        route_table.round(6),
        use_container_width=True,
        hide_index=True
    )

    st.header("4. Optimized route map")

    ordered_points = [points[0]]

    for delivery_index in best_route:
        ordered_points.append(points[delivery_index + 1])

    ordered_points.append(points[0])

    route_map = folium.Map(
        location=points[0],
        zoom_start=12,
        control_scale=True
    )

    folium.Marker(
        points[0],
        tooltip="Warehouse / Start / End",
        popup="Warehouse",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(route_map)

    for order, delivery_index in enumerate(best_route, start=1):
        point = points[delivery_index + 1]

        folium.Marker(
            point,
            tooltip=f"Visit {order}: Delivery {delivery_index + 1}",
            popup=(
                f"Visit order: {order}<br>"
                f"Delivery ID: {delivery_index + 1}<br>"
                f"Latitude: {point[0]:.6f}<br>"
                f"Longitude: {point[1]:.6f}"
            ),
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(route_map)

    folium.PolyLine(
        ordered_points,
        color="red",
        weight=4,
        opacity=0.8,
        tooltip="Optimized straight-line route"
    ).add_to(route_map)

    route_map.fit_bounds(ordered_points)

    # Stable component key and no returned map events reduce
    # unnecessary reruns caused by map interactions.
    st_folium(
        route_map,
        key="optimized_route_map",
        height=500,
        use_container_width=True,
        returned_objects=[]
    )

    st.header("5. Genetic Algorithm convergence")

    convergence_df = pd.DataFrame({
        "Generation": range(1, len(history) + 1),
        "Best distance (km)": history
    })

    st.line_chart(
        convergence_df.set_index("Generation")
    )

    st.header("6. Download results")

    export_df = pd.DataFrame([
        {
            "visit_sequence": order,
            "location": f"Delivery {delivery_index + 1}",
            "latitude": points[delivery_index + 1][0],
            "longitude": points[delivery_index + 1][1]
        }
        for order, delivery_index in enumerate(best_route, start=1)
    ])

    st.download_button(
        "Download optimized route CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="optimized_gps_route.csv",
        mime="text/csv",
        key="download_route_csv"
    )

    output_json = {
        "city": result["city"],
        "warehouse": {
            "latitude": points[0][0],
            "longitude": points[0][1]
        },
        "original_distance_km": round(baseline_distance, 4),
        "optimized_distance_km": round(optimized_distance, 4),
        "distance_reduction_percent": round(improvement, 2),
        "visit_order": visit_sequence,
        "population_size": result["population_size"],
        "generations": result["generations"],
        "distance_method": "Haversine",
        "note": "GPS demo is separate from Kaggle grid dataset."
    }

    st.download_button(
        "Download experiment results JSON",
        data=json.dumps(output_json, indent=2),
        file_name="gps_route_results.json",
        mime="application/json",
        key="download_route_json"
    )

