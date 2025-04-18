# Public Transport Network Graph Modelling using GTFS Data

This repository provides tools for processing GTFS (General Transit Feed Specification) data to model public transport networks as graphs. It includes notebooks for parsing, cleaning, and analyzing GTFS data, with support for multiple cities. Our code focuses on the Belgian data, particularly rail lines. There is also some work on the Chicago data (provided by Dr. Renzo Massobrio (renzo.massobrio@uantwerpen.be)), which we took as a reference to while working on the Belgian data.

---

## 🛠 Installation

Follow these steps to set up the project locally.  
**Note:** macOS is not supported.

### 1. Install Miniconda

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Install Build Tools

Depending on your OS:

- **Windows:**  
  Download and install the **Microsoft Visual C++ Build Tools**:  
  [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)  
  During installation, select **"Desktop Development with C++"**.

- **Linux:**  
  Install the GCC compiler:  
  ```bash
  sudo apt install build-essential
  ```
### 3. Set Up the Conda Environment
Open Anaconda Prompt (Windows) or a terminal (Linux) and navigate to the project directory:

- **Windows:**
  ```bash
  D:
  cd Documents\GTFS_TO_NETWORK
  ```
- **Linux:**
  ```bash
  cd /path/to/repo
  ```

Create the environment using:
```bash
conda env create -f environment.yml
```

### 4. Run the Jupyter Notebook
Activate the environment and start Jupyter:
```bash
conda activate pt-networks
jupyter notebook
```
In your browser, select a notebook from the list (e.g., Belgium Railways.ipynb) to begin your analysis.

## 📁 Project Structure
```python
GTFS_TO_NETWORK/
│
├── data/
│   ├── belgium/ # Unzipped GTFS files (consists of various txt files)
│   ├── belgium.zip # Raw GTFS data downloaded from the web for Belgium
│   └── gtfs_chicago.zip # Raw GTFS data downloaded from the web for Chicago
│
├── graphs/ # Consists of images of the graph at various stages of cleaning
│
├── gtfspy-master/ # Contains the GTFS processing library (GTFS-Spy) used for parsing and analyzing transit data.
│
├── notebooks/
│   ├── Belgium Railways.ipynb # Main notebook for working on the Belgian Data
│   ├── CheckNodes&Routes.ipynb # Notebook to check, visualize and analyze the L-Graph
│   ├── Chicago.ipynb # Main notebook for working on the Chicago Data
│   ├── DeleteNodes&Routes.ipynb # Notebook focussing on cleaning the L-Graph by deleting unwanted nodes and routes
│   ├── MergeRoutes.ipynb # Notebook which merges direct routes with the actual path. 
│   └── P-Space.ipynb # Notebook to work on the P-Graph being generated from the cleaned L-Graph
│   └── TopologicalIndicators_Belgium.ipynb # Notebook to work on the topological indicators of the Belgian Railways
│   └── PTopologicalIndicators_Netherlands.ipynb # Notebook to work on the topological indicators of the Dutch Railways
│
├── osmread-master/ # Includes the OSM (OpenStreetMap) reading utility for handling geographic data.
│
├── pkl/
│   ├── belgium_nodesCleaned.pkl # L-Graph after cleaning the nodes
│   ├── belgium_routesCleaned.pkl # # L-Graph after cleaning the routes after the nodes
│   ├── belgium_P.pkl # P-Graph
│   ├── belgium.pkl # Original L-Graph of Belgian Railways
│   └── chicago.pkl # Original L-Graph of Chicago Metro
│   └── gtc_data.pkl # Consists of the GTC output
│   └── nl_merged.pkl # Cleaned L-Graph of Dutch Railways
|
├── sqlite/
│   ├── belgium.sqlite 
│   └── chicago.sqlite
│
├── .gitignore
├── environment.yml
├── README.md
└── utils.py # Important functions for our work
```
