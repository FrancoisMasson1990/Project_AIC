#!/usr/bin/env python3.9
# *-* coding: utf-8 *-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copying of this file, via any medium is strictly
prohibited. Proprietary and confidential.

A library that eases the use of sqlite3 through Python
"""
import os
import sqlite3 as sql3
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def get_tablenames(sql_path):
    """Fetch table names of a SQL database.

    Parameters
    ----------
    sql_path: str
        Path to the SQL database file

    Returns
    -------
    list of str
        List of table names of the SQL database
    """
    if not Path(sql_path).exists():
        raise Exception(f"File {sql_path} does not exists")

    sqliteDB = sql3.connect(sql_path)
    sqliteDB.text_factory = lambda x: str(x, "latin1")
    sqliteCursor = sqliteDB.cursor()
    sqliteCursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in sqliteCursor.fetchall()]
    sqliteCursor.close()
    sqliteDB.close()
    return tables


def get_tablename(sql_path):
    """Fetch table name of a SQL database with a single table.

    Will return errors if multiple table names (or none) are found

    Parameters
    ----------
    sql_path: str
        Path to the SQL database file

    Returns
    -------
    str
        table name of the SQL database
    """
    if not Path(sql_path).exists():
        raise Exception(f"File {sql_path} does not exists")

    tables = get_tablenames(sql_path)
    if not len(tables):
        raise Exception(f"No table name found for file {sql_path}")
    elif len(tables) > 1:
        raise Exception(
            f"Multiple table names found for file {sql_path}: " f"{tables}"
        )
    tablename = tables[0]
    return tablename


def get_version(sql_path):
    """Fetch version of a SQL database.

    The version needs to have been entered in PRAGMA user_version.
    This is the case for our sorted_databases.

    Parameters
    ----------
    sql_path: str
        Path to the SQL database file

    Returns
    -------
    int
        version of the database
    """
    if not Path(sql_path).exists():
        raise Exception(f"File {sql_path} does not exists")

    sqliteDB = sql3.connect(sql_path)
    sqliteDB.text_factory = lambda x: str(x, "latin1")
    sqliteCursor = sqliteDB.cursor()
    version = sqliteCursor.execute("PRAGMA user_version").fetchall()
    sqliteCursor.close()
    sqliteDB.close()

    if not len(version):
        raise Exception(f"WARNING: Version not found for file {sql_path}")
    elif len(version) > 1:
        raise Exception(f"WARNING: Multiple versions for file {sql_path}")
    version = version[0][0]
    return version


def get_columns(sql_path):
    """Return column names of a SQL database.

    Parameters
    ----------
    sql_path: str
        Path to the SQL database file

    Returns
    -------
    list of str
        Column names from database
    """
    if not Path(sql_path).exists():
        raise Exception(f"File {sql_path} does not exists")

    sqliteDB = sql3.connect(sql_path)
    sqliteDB.text_factory = lambda x: str(x, "latin1")
    sqliteCursor = sqliteDB.cursor()
    columns = sqliteCursor.execute("PRAGMA table_info(data)").fetchall()
    sqliteCursor.close()
    sqliteDB.close()

    columns = [c[1] for c in columns]
    return columns


def get_count(sql_path):
    """Return number of lines of a SQL database.

    Supposes that column rowid is counting the rows, starting from 1.

    Parameters
    ----------
    sql_path: str
        Path to the SQL database file

    Returns
    -------
    int
        Number of lines of the database
    """
    if not Path(sql_path).exists():
        raise Exception(f"File {sql_path} does not exists")

    sqliteDB = sql3.connect(sql_path)
    sqliteDB.text_factory = lambda x: str(x, "latin1")
    sqliteCursor = sqliteDB.cursor()
    tablename = get_tablename(sql_path)

    count_cmd = "SELECT MAX(rowid) from " + tablename
    count = sqliteCursor.execute(count_cmd).fetchall()
    sqliteCursor.close()
    sqliteDB.close()

    if not len(count):
        raise Exception(f"WARNING: n_lines not found for file {sql_path}")
    elif len(count) > 1:
        raise Exception(f"WARNING: Multiple n_lines for file {sql_path}")
    count = count[0][0]

    if count is None:
        count = 0

    return count


def get_count_values(sql_path, column):
    """Return number of unique value in a specific column of a SQL database.

    Parameters
    ----------
    sql_path: str
        Path to the SQL database file

    column: str
        Name of the column present in the SQl database

    Returns
    -------
    df: pandas Dataframe
        dataframe where first column is the unique value
        and the second column the number of occurence
    """
    if not Path(sql_path).exists():
        raise Exception(f"File {sql_path} does not exists")

    sqliteDB = sql3.connect(sql_path)
    sqliteDB.text_factory = lambda x: str(x, "latin1")
    sqliteCursor = sqliteDB.cursor()
    tablename = get_tablename(sql_path)

    cols = get_columns(sql_path)
    if column not in cols:
        raise Exception(f"WARNING: No such column :{column} in {sql_path}")

    count_cmd = (
        f"SELECT {column}, COUNT(*) "
        + f"from {tablename} "
        + f"GROUP BY {column}"
    )

    count = sqliteCursor.execute(count_cmd).fetchall()
    sqliteCursor.close()
    sqliteDB.close()

    if not len(count):
        raise Exception(f"WARNING: empty database: {sql_path}")

    df = pd.DataFrame(count, columns=[column, "counts"])

    return df


def from_sql(
    sql_path, columns=None, conditions=None, chunk_size=None, verbose=False
):
    """Import a .db SQL database into pandas.

    This function returns an iterator of chunks of the database.
    Therefore, it must be used as
        for df in from_sql(sql_path):
            ...

    Parameters
    ----------
    sql_path : str
        Path to the SQL database file

    columns : list of str, optional
        List of columns to be returned.
        Default is None, which returns all columns

    conditions: list of str, optional
        List of conditions to inject in the SQL command.
        i.e. ['event_description = "WEB_RUNTIME']
        You can also use a filters.py:KeepValues() object as a condition,
        All conditions are treated as AND
        Default is None, which returns all lines.

    chunk_size: int, optional
        If not None, will only return [CHUNK_SIZE] rows at a time

    verbose: bool, optional
        If True, will print tqdm progress bar on screen.
        Default is False

    Returns
    -------
    iterator of pandas dataframe
    """
    sql_path = os.path.abspath(os.path.expanduser(sql_path))
    if not Path(sql_path).exists():
        raise Exception(f"File {sql_path} does not exists")

    # version = get_version(sql_path)

    try:
        tablename = get_tablename(sql_path)
    except sql3.DatabaseError:
        print(f"ERROR: {sql_path} DATABASE DISK IMAGE IS MALFORMED")
        return []

    count = get_count(sql_path)

    if columns is None:
        cmd_template = f"SELECT * from {tablename}"
        columns = get_columns(sql_path)
    else:
        if isinstance(columns, str):
            columns = [columns]
        cmd_template = f"SELECT {' ,'.join(columns)} from {tablename}"

    if conditions is None:
        conditions = []
    if not isinstance(conditions, list) and not isinstance(conditions, set):
        conditions = [conditions]
    for i_c, c in enumerate(conditions):
        if hasattr(c, "sql_condition") and c.sql_condition:
            conditions[i_c] = c.sql_condition

    sqliteDB = sql3.connect(sql_path)
    sqliteDB.text_factory = lambda x: str(x, "latin1")

    if chunk_size is None:
        chunk_size = count
    if count:
        n_chunks = (count - 1) // chunk_size + 1
    else:
        n_chunks = 1

    if verbose:
        pbar = tqdm(total=n_chunks)
    for i_chunk in range(n_chunks):
        offset = i_chunk * chunk_size + 1
        chunk_limit = f"(rowid >= {offset} AND rowid < {offset+chunk_size})"
        chunk_conditions = conditions + [chunk_limit]
        cmd_chunk = f'{cmd_template} WHERE ({" AND ".join(chunk_conditions)})'

        output = pd.read_sql_query(cmd_chunk, sqliteDB)
        output.reset_index(inplace=True, drop=True)
        if len(output):
            yield output
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
    sqliteDB.close()


def load_sql(sql_path, **kwargs):
    """Import a .db SQL database into pandas.

    This function will import the whole file in RAM.
    For a more precise importation, use from_sql() instead.

    Parameters
    ----------
    sql_path : str
        Path to the SQL database file

    Returns
    -------
    pandas dataframe
    """
    df = pd.DataFrame()
    for df in from_sql(sql_path, **kwargs):
        pass
    return df


def delete_sql(sql_path):
    """Delete all data from a SQL file to only keep the skeleton.

    Will delete all data to the file on disk. Use carefully.

    Parameters
    ----------
    sql_path : str
        Path to the SQL database file
    """
    tablename = get_tablename(sql_path)

    sqliteDB = sql3.connect(sql_path)
    sqliteDB.text_factory = lambda x: str(x, "latin1")
    sqliteCursor = sqliteDB.cursor()
    sqliteCursor.execute(f"DELETE from {tablename};")
    sqliteDB.commit()
    sqliteCursor.execute("VACUUM")
    sqliteDB.commit()
    sqliteCursor.close()
    sqliteDB.close()


def insert_sql(df, sql_path):
    """Insert data to a SQL file.

    Will insert data directly on the file on disk. Use carefully.

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe containing the data to insert.
        Must contain all columns from the v4 sorted database format.

    sql_path : str
        Path to the SQL database file
    """
    col_list = get_columns(sql_path)

    tablename = get_tablename(sql_path)

    sqliteDB = sql3.connect(sql_path)
    sqliteDB.text_factory = lambda x: str(x, "latin1")
    sqliteCursor = sqliteDB.cursor()

    for _, row in df.iterrows():
        cols = []
        data = []
        for col in col_list:
            if col in df:
                cols.append(col)
                data.append(row[col])
        if not cols:
            continue
        data_tuple = tuple(data)
        cmd = (
            f'INSERT INTO {tablename} ({",".join(cols)}) '
            f'VALUES ({", ".join(["?"]*len(cols))});'
        )
        sqliteCursor.execute(cmd, data_tuple)
    sqliteDB.commit()
    sqliteCursor.close()
    sqliteDB.close()


def to_sql(df, sql_path, name="data"):
    """
    Save a pandas dataframe to a SQL .db dataframe.

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe containing the data to insert.
        Must contain all columns from the v4 sorted database format.

    sql_path : str
        Path to the SQL database file to save
    """
    columns = tuple(df.columns)
    query = f"CREATE TABLE IF NOT EXISTS {name} {columns}"
    conn = sql3.connect(sql_path)
    c = conn.cursor()
    c.execute(query)
    conn.commit()
    df.to_sql(name, conn, if_exists="replace", index=False)
