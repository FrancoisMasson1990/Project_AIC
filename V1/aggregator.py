#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Aggregate data into a single database coming from both
upcoming and top collections on a daily basis
"""

import pandas as df
import top_aggregator as top
import upcoming_aggregator as up
import datetime


if __name__ == "__main__":
    date = datetime.date.today()
    upcomings_list = up.get_upcomings()
    upcomings_list.insert(loc=0,
                          column="date",
                          value=date.strftime('%Y-%m-%d'))
    print(upcomings_list)
