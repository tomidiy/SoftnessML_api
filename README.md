# SoftnessML API
$$
a^2 + b^2 = c^2
$$

A machine learning framework for predicting local rearrangement propensity ("softness") in supercooled liquids and glassy systems using structure-based descriptors and Support Vector Machine (SVM) models.

## Table of Contents
- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Mathematical Framework](#mathematical-framework)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Applications](#applications)
- [References](#references)


## Overview
**SoftnessML API** is a Python-based framework that uses machine learning to predict the "softness" of particles in disordered systems such as supercooled liquids and glasses. Softness is a structure-based quantity that correlates strongly with the local propensity for particle rearrangements, enabling predictions of dynamics from static structural information.

This project provides:
- Calculation of local structural descriptors (radial, angular, and bond-orientational order parameters via spherical harmonics) from molecular dynamics (MD) trajectories stored in GSD files
- Trained Support Vector Machine (SVM) model for softness prediction
- A FastAPI-based RESTful API for easy integration into simulation pipelines
- Efficient computation using Numba acceleration

The implementation follows the original SVM methodology introduced by Schoenholz et al. (2016).

## Scientific Background
### What is Softness?
In supercooled liquids and glasses, particle dynamics are highly heterogeneous. Some particles rearrange frequently, while others remain relatively static. **Softness** is a machine learning-derived scalar field that captures the local structural environment of each particle and predicts its likelihood to undergo a rearrangement.

The concept was introduced by Schoenholz et al. (2016) and has become a widely used tool for linking structure to dynamics in amorphous materials.

### Why Predict Rearrangements?
Predicting particle rearrangements has important implications in:
- **Fundamental Physics**: Insights into the glass transition and dynamic heterogeneity
- **Materials Science**: Design of stable amorphous materials and understanding of mechanical properties
- **Industrial Applications**: Prediction of aging, failure, and deformation in glassy systems

## Mathematical Framework
Detailed theoretical background and mathematical formulations
used in this project are available in the GitHub Wiki:

📐 **Mathematical Framework**  
https://github.com/tomidiy/SoftnessML_api/wiki/Mathematical-Framework


## Installation
### Prerequisites
- Python 3.9+
- Docker (recommended for deployment)

### From Source
```bash
git clone https://github.com/tomidiy/SoftnessML_api.git
cd SoftnessML_api
```

## Dependencies
See `app/requirements.txt`:
- fastapi==0.115.0
- uvicorn==0.30.6
- numpy==1.26.4
- scikit-learn==1.5.1
- gsd==3.3.0
- pydantic==2.8.2
- scipy==1.13.1
- numba==0.60.0

Install locally (for development):
```bash
pip install -r app/requirements.txt
```


## Usage
**Prepare Data Files**:
Place the following files in the `data/` directory:
-`Softness_train_data.pkl`
-`Softness_train_data_radialSphericalHarmonics.pkl` 
-`phop_T<temp>.pkl`
-`T<temp>_idx0.gsd`


**Run with Docker (Recommended)**
```bash
docker build -t softness-predictor .
docker run -d -p 8000:8000  \
     -v $(pwd)/data:/app/data \
      --name softness-predictor-container  \
      softness-predictor
```

**Health Check**:

```bash
curl http://localhost:8000/health
```

Expected output: {"status": "healthy"}

**Predict Softness**:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"temp": 0.7, "frame": 0, "gsd_file": "T0.7_idx0.gsd"}'
```

## API Reference
`/health`
     - GET — Returns service status
`/predict`
     - POST — Computes structural descriptors (radial, angular, and spherical harmonics-based bond-orientational parameters) from the specified GSD file and frame, then predicts softness using the trained SVM.
     Request body:

```json
{
  "temp": 0.7,
  "frame": 0,
  "gsd_file": "T0.7_idx0.gsd"
}
```
Response:
Array of softness values for all particles in the selected frame.

## Project Structure
```text
SoftnessML_api/
├── app/
│   ├── main.py
│   ├── model.py
│   ├── Structure.py
│   ├── requirements.txt
├── data/
│   ├── Softness_train_data.pkl
│   ├── Softness_train_data_radialSphericalHarmonics.pkl
│   ├── phop_T<temp>.pkl
│   └── T<temp>_idx0.gsd
├── Dockerfile
├── README.md
└── .github/workflows/ci.yml
```

## Applications
- Understanding dynamic heterogeneity and the glass transition
- Predicting shear transformation zones in metallic and colloidal glasses
- Identifying structurally mobile regions in amorphous pharmaceuticals
- Analyzing structural origins of plasticity and failure in disordered solids
- Benchmarking structure–dynamics correlations in simulation datasets

## References
- Schoenholz, S. S., et al. (2016). A structural approach to relaxation in glassy liquids. Nature Physics, 12, 469–471.
- Cubuk, E. D., et al. (2015). Identifying structural flow defects in disordered solids using machine-learning methods. Physical Review Letters, 114, 108001.
- Steinhardt, P. J., Nelson, D. R., & Ronchetti, M. (1983). Bond-orientational order in liquids and glasses. Physical Review B, 28, 784.


