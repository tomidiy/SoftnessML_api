# SoftnessML_api

A FastAPI-based web service for predicting particle softness in molecular dynamics simulations
using machine learning (Softness methodology). The API processes GSD files to compute softness based on radial and angular 
structural features.


## Project Structure
```
SoftnessML_api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── model.py             # Softness prediction model
│   ├── Structure.py         # Radial and angular structure calculations
│   ├── requirements.txt     # Python dependencies
├── data/                    # Data files 
│   ├── Softness_train_data.pkl
│   ├── Softness_train_data_radialSphericalHarmonics.pkl
│   ├── phop_T<temp>.pkl
│   ├── T<temp>_idx0.gsd
├── Dockerfile               # Docker configuration
├── README.md               # This file
└── .github/workflows/ci.yml # CI/CD configuration
```

## Prerequisites
- Docker
- Python 3.9 (for local development)
- Data files in data/ directory

## Setup
- **Clone the Repository**:

```bash
git clone https://github.com/tomidiy/SoftnessML_api.git
cd SoftnessML_api
```

- **Prepare Data Files**:
Ensure `Softness_train_data.pkl`, `Softness_train_data_radialSphericalHarmonics.pkl`, `phop_T<temp>.pkl`, and `T<temp>_idx0.gsd`
are in the `data/` directory.

- **Build and Run Docker Container**:

```bash
docker build -t softness-predictor .
docker run -d -p 8000:8000 -v $(pwd)/data:/app/data --name softness-predictor-container softness-predictor
```

## Usage
- **Health Check**:

```bash
curl http://localhost:8000/health
```

Expected output: {"status": "healthy"}

- **Predict Softness**:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"temp": 0.7, "frame": 0, "gsd_file": "T0.7_idx0.gsd"}'
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


## Author
Tomilola Obadiya

## License
This project is licensed under the MIT License.

