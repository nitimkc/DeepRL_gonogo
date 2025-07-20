
# import 

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# YOUR_PATH = "/gscratch5/users/nmishra/DeepRL_gonogo/data"
# savepath = Path(YOUR_PATH)
# print(savepath)

print('creating parser')
parser = argparse.ArgumentParser(description="Twitter ILI infection detection with LLMs")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
args = parser.parse_args()

ROOT = Path(args.root)
DATA = ROOT.joinpath(f"data")
DATA.mkdir(parents=True, exist_ok=True)

boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
data_set = boc.get_ophys_experiment_data(501498760)
scenes = data_set.get_stimulus_template('natural_scenes')

session_images_nums = [37, 112, 13, 84, 79, 48, 45, 116]
scenes = scenes[session_images_nums]
for idx, s in zip(session_images_nums, scenes):
  print(idx)
  # plt.imshow(s, cmap='gray')
  plt.savefig(DATA.joinpath(f"img_{idx}.png"))
  # plt.show()