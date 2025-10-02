import os
import pickle
import gsd.hoomd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from typing import Dict, List, Tuple
from Structure import get_structure, get_radial_and_angular_structure

# Define the base directory as the directory containing this script
APP_DIR = Path(__file__).resolve().parent

class SoftnessPredictor:
    def __init__(self, radial_only: bool = False):
        self.radial_only = radial_only
        self.clf, self.mean, self.std = self._load_classifier()
        self.sigma = 0.2
        self.particles_A = np.arange(3277, dtype=int)

    def _load_classifier(self) -> Tuple[SVC, np.ndarray, np.ndarray]:
        """Load and return the trained SVM classifier with mean and std for scaling."""
        if self.radial_only:
            train_file = f'Softness_train_data.pkl'
            with open(os.path.join(APP_DIR.parent/'data', train_file), 'rb') as f:
                training_data = pickle.load(f)
        else:
            train_file = f'Softness_train_data_radialSphericalHarmonics.pkl'
            with open(os.path.join(APP_DIR.parent/'data', train_file), 'rb') as f:
                training_data = pickle.load(f)

        X_train = training_data['X_train_rawdata']
        y_train = training_data['Y_train']
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_scaled = (X_train - mean) / std
        clf = SVC(C=0.01, kernel='linear').fit(X_scaled, y_train)
        return clf, mean, std

    def _load_snapshot(self, filename: str, frame: int) -> Dict:
        """Load a snapshot from a .gsd file."""
        data = {'Positions': [], 'Box_size': []}
        with gsd.hoomd.open(os.path.join(APP_DIR.parent/'data', filename), 'r') as snapshots:
            data['Box_size'] = snapshots[frame].configuration.box[:3]
            data['Positions'] = snapshots[frame].particles.position
        return data

    def _load_phop(self, temp: float, frame: int) -> np.ndarray:
        """Load phop data for the given temperature and frame."""
        phop_file = f'phop_T{temp}.pkl'
        with open(os.path.join(APP_DIR.parent/'data', phop_file), 'rb') as f:
            phop_data = pickle.load(f)
        return phop_data['phop'][:, frame + 2]

    def predict(self, temp: float, frame: int, gsd_file: str) -> Dict:
        """Predict softness for a given temperature, frame, and snapshot file."""
        _max = 0.2              #threshold for rearrangement
        snapshot = frame + 5     #offset for phop calculation
        temp_data = self._load_snapshot(gsd_file, snapshot)
        position = np.array(temp_data['Positions'])
        box = np.array(temp_data['Box_size'])
        phop = self._load_phop(temp, snapshot)

        # Compute structures
        if self.radial_only:
            structures = [get_structure(p, position, box, self.sigma) for p in self.particles_A]
        else:
            structures = [get_radial_and_angular_structure(p, position, box, self.sigma) for p in self.particles_A]

        structures = (np.array(structures) - self.mean) / self.std
        softness = self.clf.decision_function(structures)

        # Conditioned softness on rearrangements
        vals = np.where(phop > _max)[0]
        cond_soft = [softness[val] for val in vals] if vals.size > 0 else []

        return {
            'softness': softness.tolist(),
            'SconditionedonR': cond_soft
        }