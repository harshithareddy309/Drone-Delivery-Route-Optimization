# 🚁 Drone Delivery Route Optimization Using Genetic Algorithm

## 📌 Project Overview
This project optimizes the visiting order of multiple drone delivery locations using a Genetic Algorithm (GA). It uses GPS coordinates and the Haversine formula to estimate the total straight-line distance travelled from a warehouse through all delivery locations and back to the warehouse.

The application is developed using Python and Streamlit, with an interactive map for route visualization.

## 🎯 Objectives
- Optimize the order of drone delivery locations.
- Compare the original route with the optimized route.
- Visualize the route on an interactive map.
- Display the distance reduction and algorithm convergence.
- Export route results in CSV and JSON formats.

## 🛠️ Technologies Used
- Python
- Genetic Algorithm
- Streamlit
- Folium
- Streamlit-Folium
- NumPy
- Pandas
- Haversine Distance Formula
- Kaggle

## ✨ Features
- Select from multiple city presets.
- Enter warehouse and delivery GPS coordinates.
- Configure population size and number of generations.
- Calculate original and optimized route distances.
- View the optimized delivery sequence.
- Visualize the route on an interactive map.
- View a Genetic Algorithm convergence graph.
- Download route details as CSV and JSON files.
- View details of the Kaggle grid-based experiment.

## ⚙️ Installation and Execution

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder
```bash
cd YOUR_REPOSITORY_NAME
```

### 3. Install dependencies
```bash
py -m pip install -r requirements.txt
```

### 4. Run the application
```bash
py -m streamlit run app.py
```

### 5. Use the application
Open the local address displayed in the terminal, enter the warehouse and delivery coordinates, and click **Optimize Delivery Route**.

## 🧬 Algorithm
The Genetic Algorithm uses:
1. Initial population generation.
2. Route-distance evaluation.
3. Tournament selection.
4. Order crossover.
5. Mutation through swapping delivery positions.
6. Elitism to preserve strong candidate routes.
7. Repeated generations to search for a shorter route.

## 📏 Distance Calculation
The Haversine formula estimates the great-circle distance between two geographic coordinates.

**Note:** The displayed distances are straight-line estimates. They do not account for buildings, restricted airspace, drone flight regulations, battery constraints, or approved flight corridors.

## 📊 Experimental Dataset
The project also includes a Kaggle-based drone delivery experiment using grid-coordinate data. The grid-based experiment and the GPS visualization are separate experiments.

## 👩‍💻 Project Purpose
Developed as a Soft Computing case study to demonstrate Genetic Algorithm-based optimization and interactive route visualization.