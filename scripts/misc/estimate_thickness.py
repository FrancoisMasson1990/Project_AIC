#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Main script to estiamte valve thickness.
"""

import numpy as np
import pandas as pd

import aic.misc.files as fs
import aic.misc.sql as sql
import aic.processing.operations as op

if __name__ == "__main__":
    layers = [
        19,
        21,
        23,
        25,
        29,
    ]
    dfs = []
    threshold = 900
    for j in layers:
        # load projected native valve
        path = (
            fs.get_native_root() / f"Magna/projected/Magna_{j}/projected.npy"
        )
        with open(str(path), "rb") as f:
            native = np.load(f)

        valve = native.copy()
        # Project along z axis to help for circle-points fitting
        valve_p = valve.copy()
        valve_p[:, 2] = np.round(valve_p[:, 2])
        valve_threshold = valve_p.copy()
        valve_threshold = valve_p[valve_p[:, 3] > threshold]
        candidates = np.array([False] * valve.shape[0])
        # For each layer, attempt to fit a circle using the component
        # of the metalic part and remove points outside of it by saving
        # its index position for the last column
        print(f"Size {j}")
        df = []
        for i, z in enumerate(np.unique(valve_p[:, 2])):
            valve_threshold_z = valve_threshold[valve_threshold[:, 2] == z]
            if valve_threshold_z.shape[0] > 2:
                xc, yc, r, _ = op.leastsq_circle(
                    valve_threshold_z[:, 0], valve_threshold_z[:, 1]
                )
                circle_center = np.array([xc, yc, z])
                layer = []
                for point in valve_p[valve_p[:, 2] == z]:
                    dist = op.euclidean(point[:3], circle_center)
                    layer.append(dist)
                df.append([z, r - min(np.asarray(layer))])

        df = np.array(df)
        df = pd.DataFrame(df, columns=["layer", "thickness"])
        df["size"] = j
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs = dfs[["size", "layer", "thickness"]]
    dfs.reset_index(drop=True, inplace=True)
    sql_path = str(fs.get_native_root() / "Magna" / "thickness_info.db")
    sql.to_sql(dfs, sql_path)
