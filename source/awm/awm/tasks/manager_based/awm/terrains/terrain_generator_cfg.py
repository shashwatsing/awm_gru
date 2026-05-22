# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.00, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.00, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.0, 0.25), num_waves=4, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.0, 0.08), noise_step=0.02, border_width=0.25
        ),
    },
)

# Evaluation-only terrain: pyramid stairs, visible from above as a raised pyramid.
# Robot spawns at the top platform and descends — leg extension visible on the steps.
# Use with Template-Awm_StairsEval-v0 and num_envs=4 or 8.
STAIRS_EVAL_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=5,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.00, 0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# Unseen eval terrain 1: pure wave — undulating hills not present in training distribution.
# High-amplitude wave eval: amplitude 0.20–0.45m, 6 waves per tile.
# Larger amplitude + more waves = steeper local gradients → triggers roughness threshold
# and forces visible leg extension. Training used 0–0.25m amplitude, 4 waves.
WAVE_EVAL_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=5,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.10, 0.30),
            num_waves=4,
            border_width=0.25,
        ),
    },
)

# Unseen eval terrain 2: pure random rough — dense random bumps not seen as a dedicated
# terrain during training (training mix included 20% rough but always mixed with flat/stairs).
# noise_range 0.04–0.12m is harder than the training 0–0.08m upper end.
ROUGH_EVAL_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=5,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.04, 0.12),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)


