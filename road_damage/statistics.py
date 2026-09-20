"""Source-labelled paper results and local training-log readers."""
import csv
import json
import math
from pathlib import Path

PAPER_URL = 'https://doi.org/10.1038/s41598-024-62933-z'
PAPER_COMPARISON = [
    {'Model': 'YOLOv8n', 'mAP@50 (%)': 69.2, 'mAP@50–95 (%)': 38.0,
     'F1': .67, 'Parameters (M)': 3.1, 'GFLOPs': 8.2},
    {'Model': 'YOLOv8-PD', 'mAP@50 (%)': 70.6, 'mAP@50–95 (%)': 39.5,
     'F1': .68, 'Parameters (M)': 2.3, 'GFLOPs': 6.1},
]
PAPER_CLASSES = [
    {'Damage': 'D00 · Longitudinal crack', 'YOLOv8n AP@50 (%)': 74.2, 'YOLOv8-PD AP@50 (%)': 76.2},
    {'Damage': 'D10 · Transverse crack', 'YOLOv8n AP@50 (%)': 79.3, 'YOLOv8-PD AP@50 (%)': 79.7},
    {'Damage': 'D20 · Alligator crack', 'YOLOv8n AP@50 (%)': 71.9, 'YOLOv8-PD AP@50 (%)': 73.3},
    {'Damage': 'D40 · Pothole', 'YOLOv8n AP@50 (%)': 51.2, 'YOLOv8-PD AP@50 (%)': 53.1},
]
MODULES = [
    {'Component': 'BOT', 'Role in the paper': 'Capture long-range relationships and global crack features',
     'Local PD-inspired implementation': 'Residual multi-head attention after backbone layer 8'},
    {'Component': 'LSKA', 'Role in the paper': 'Emphasize relevant damage features with large-kernel attention',
     'Local PD-inspired implementation': 'Separable depthwise kernels and residual sigmoid gating'},
    {'Component': 'C2fGhost', 'Role in the paper': 'Reduce neck computation while extracting damage features',
     'Local PD-inspired implementation': 'GhostBottleneck inside neck C2f layers 12, 15, 18 and 21'},
    {'Component': 'LSCD-Head', 'Role in the paper': 'Share convolutions and use GroupNorm for a lightweight head',
     'Local PD-inspired implementation': 'Shared 64-channel head with per-scale regression multipliers'},
]


def read_run(folder):
    folder = Path(folder)
    rows = []
    path = folder / 'results.csv'
    if path.is_file():
        with path.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                parsed = {}
                for key, value in row.items():
                    if key is not None:
                        try:
                            numeric = float(value)
                            if math.isfinite(numeric):
                                parsed[key.strip()] = numeric
                        except (TypeError, ValueError):
                            pass
                if 'epoch' in parsed and parsed['epoch'] >= 1 and parsed['epoch'].is_integer():
                    rows.append(parsed)
    metadata_path = folder / 'experiment.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError('Experiment metadata must be a JSON object')
    return {'run': folder.name, 'history': rows, 'configuration': metadata,
            'checkpoint': str(folder / 'weights/best.pt'),
            'checkpoint_exists': (folder / 'weights/best.pt').is_file()}
