# SoftnessML API

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
### Local Structure Descriptors
The local environment of each particle is characterized by radial structure functions, angular structure functions, and bond-orientational order parameters computed via spherical harmonics.

1. **Radial Structure Functions**  
   Capture local density variations at different distances:

$$
G_\mu(i) = \sum_{j \in \mathrm{neighbors}} e^{-(r_{ij} - \mu)^2 / L^2}
$$

   where $\mu$ are radial distance parameters and $L$ is a characteristic length scale.

2. **Angular Structure Functions**  
   Capture three-body correlations:

$$
\Psi_{\xi,\lambda,\zeta}(i) = \sum_{j,k \in \text{neighbors}} e^{-\xi^2 (r_{ij}^2 + r_{ik}^2 + r_{jk}^2)} (1 + \lambda \cos \theta_{jik})^\zeta
$$

   where $\theta_{jik}$ is the angle at particle $i$ formed by neighbors $j$ and $k$, and $\xi$, $\lambda$, $\zeta$ control radial decay and angular sensitivity.

3. **Bond-Orientational Order Parameters (Steinhardt Parameters)**  
   These quantify the degree of local rotational symmetry using spherical harmonics. For a central particle $i$, neighbors within annular  shells (defined by inner radii $r_{\text{inner}}$ and width $\Delta r = 0.5$) are considered.

   For each shell and each even angular momentum $l$ (typically $l = 2, 4, 6, 8, 10, 12, 14$):

$$
q_{lm}(i) = \frac{1}{N_b(i)} \sum_{j \in \text{shell}} Y_{lm}(\theta_{ij}, \phi_{ij})
$$

$$
q_l(i) = \sqrt{\frac{4\pi}{2l + 1} \sum_{m=-l}^{l} |q_{lm}(i)|^2}
$$

   where:
   - $N_b(i)$ is the number of neighbors in the shell
   - $Y_{lm}(\theta, \phi)$ are spherical harmonics
   - $(\theta_{ij}, \phi_{ij})$ are the polar and azimuthal angles of the vector from particle $i$ to neighbor $j$

   The rotationally invariant $q_l(i)$ measures the strength of $l$-fold symmetry in that shell. In practice, the implementation computes the average of $|Y_{lm}|$ over neighbors and then applies the normalization (equivalent under magnitude).

### Softness Calculation (SVM)
Softness is computed as a linear combination of the structural descriptors using a trained Support Vector Machine: Softness is 

$$
S_i = \mathbf{w} \cdot \mathbf{x}_i + b
$$

where:
- $\mathbf{x}$ is the concatenated feature vector (radial $G_\mu$ and bond-orientational $q_l$ or angular $\Psi_{\xi,\lambda,\zeta}$ for multiple shells) for particle $i$
- $\mathbf{w}$ is the learned weight vector
- $b$ is the bias term

The SVM is trained to separate particles that undergo significant motion. To quantify particle motion, we use the **hop parameter** $p_{\text{hop}}$, following the activated‑dynamics framework of Candelier et al. This metric measures particle displacement over a fixed observation window.

For a window of 10 Lennard‑Jones time units:

$$
p_{\text{hop}}(i,t) =
\sqrt{
\left\langle
\left(\vec{r}_i(t) - \langle \vec{r}_i \rangle_{w_2}\right)^2
\right\rangle_{w_1}
\left\langle
\left(\vec{r}_i(t) - \langle \vec{r}_i \rangle_{w_1}\right)^2
\right\rangle_{w_2}
}
$$

where the time windows are
$w_1 = [t-5, t]$ and $w_2 = [t, t+5]$, and
$\langle \cdot \rangle_{w_i}$ denotes an average over the corresponding half‑window.




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


