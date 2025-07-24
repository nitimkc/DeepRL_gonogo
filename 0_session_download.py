
from pathlib import Path
import logging
import argparse

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
import pathlib
import pandas as pd
import numpy as np

print('creating parser')
parser = argparse.ArgumentParser(description="DESCRIPTION HERE")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
args = parser.parse_args()

ROOT = Path(args.root)
DATA = ROOT.joinpath("data")
SESSION_DATA = DATA.joinpath(f"session")
SESSION_DATA.mkdir(parents=True, exist_ok=True)

# Set up cache
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=SESSION_DATA)

# Load the session table
sessions = cache.get_session_table()
print(f"Total sessions found: {len(sessions)}")

# Download all .nwb files
for session_id in sessions.index:
    print(f"Downloading session {session_id}...")
    session = cache.get_session_data(session_id)
    print(f"Saved NWB file: {session.nwb_file_path}")