# =============================================================================
# Originally from ASSET (AlabamaASRL/asset_asrl)
# Copyright 2020-present The University of Alabama-Astrodynamics and Space
#   Research Lab. Licensed under the Apache License, Version 2.0
# License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# Original Developer: James B. Pezent
#
# Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
#   Apache 2.0 — see LICENSE.txt):
#   - Package renamed: asset_asrl -> tycho
#   - Module renamed: asset_asrl (pybind11) -> _tycho (nanobind)
#   - Imports updated accordingly
# =============================================================================

import csv
import os

import numpy as np

import tycho.Astro.Constants as c

norm = np.linalg.norm


def WriteData(traj, name, Folder="Data"):
    if Folder != "":
        if not os.path.exists(Folder + "/"):
            os.makedirs(Folder + "/")
    np.save(Folder + "/" + name + ".npy", traj)


def ReadData(name, Folder="Data"):
    return np.load(Folder + "/" + name + ".npy", allow_pickle=True)


def ReadCopernicusFile(name, Folder="Data"):
    StateHist = []
    with open(Folder + "/" + name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(readCSV):
            if i > 4:
                S = np.zeros((8))
                for j in range(0, 6):
                    S[j] = float(row[4 + j]) * 1000.0
                S[6] = float(row[3]) * c.day
                S[7] = float(row[1])
                StateHist.append(S)
    return StateHist
