# Isaac Lab 3.0 port verification run

`2026-09-05_mappo_flycrane_512envs_isaaclab3.csv` is the `Reward / Total reward (mean)` curve
from the run used to verify the Isaac Lab 3.0 / Isaac Sim 6.0.1 / skrl 2.1.0 port of
`Isaac-flycrane-payload-decentralized-hovering-v0`.

    python scripts/skrl/train.py --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
        --headless --num_envs=512 --seed=42 --algorithm=MAPPO --max_iterations=1172

| | this run | published baseline |
|---|---|---|
| envs | 512 | 4096 |
| timesteps | 150,000 | 383,900 |
| samples | ~77M | ~1.57B |
| final reward | 176.6 | 230.6 |
| peak reward | 207.4 | 244.9 |
| wall clock | 2.2 h | 19.1 h |

512 is the ceiling on an 8 GB RTX 5070 Laptop: 1024 envs runs out of memory inside the MAPPO
update, not the simulation. With ~20x fewer samples the run does not reach the baseline plateau,
so this is a port-correctness check, not a reproduction. The curve is noisier than the baseline's
because 512 envs gives noisier gradients; it dips around 100k and 140k and recovers both times.
