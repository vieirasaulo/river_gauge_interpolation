from typing import Union
from pathlib import Path

import numpy as np
from shapely import (
    Point,
    MultiPoint,
    LineString,
)

from shapely.ops import split
import pandas as pd
from pandas import Timestamp, DataFrame
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib
import matplotlib.pyplot as plt


def remove_z_coordinate(in_shp, out_shp, save_file=True):
    """function to remove the z coordinate from points
    return 3 main points
    """
    in_gdf = gpd.read_file(in_shp)
    out_gdf = in_gdf.copy()

    new_geometry = [Point(list(point.coords)[0][:2]) for point in in_gdf.geometry]
    out_gdf["geometry"] = new_geometry  # all of the points are here, without Z

    out_gdf_dropped = (
        out_gdf.drop_duplicates(subset="geometry")
        .drop_duplicates(subset="StationNam")
        .reset_index(drop=True)
    )
    if save_file:
        out_gdf_dropped.to_file(out_shp)

    return out_gdf_dropped


def split_line(in_line: LineString, in_points: MultiPoint):

    pts = [point for point in in_points.geoms]

    pts = [in_line.interpolate(in_line.project(pt)) for pt in pts]
    out_pts = MultiPoint(pts)
    out_line = split(in_line, out_pts)

    return out_line, out_pts


def interpolate_by_distance(
    in_line: LineString, split_distance: Union[int, float]
) -> MultiPoint:
    """Function to get populate a shapely Linestring with points.

    Args:
        in_line (LineString): geometry one wishes to split.
        split_distance (Union[int, float]): distance between points.

    Returns:
        Points over the line (MultiPoint)
    """

    line = in_line
    space = split_distance
    n_points = int(line.length / space)  # that will split the lines
    points_list = [
        line.interpolate((i / n_points), normalized=True) for i in range(1, n_points)
    ]
    # append extreme points
    if isinstance(line, LineString):
        points_list.append(Point(list(line.coords)[0]))
        points_list.append(Point(list(line.coords)[-1]))

    points = MultiPoint(points_list)
    return points


def points_from_line(ref_path, out_shp, split_distance, save_file=True):
    """converts reference_line_river with unjoined lines into points"""

    reference_line = gpd.read_file(ref_path)
    crs = reference_line.crs
    line = reference_line.unary_union
    # multipoints
    mps = interpolate_by_distance(in_line=line, split_distance=split_distance)

    out_gdf = (
        gpd.GeoDataFrame(geometry=[mps], crs=crs)
        .explode(index_parts=True)
        .reset_index(drop=True)
    )

    if save_file:
        out_gdf.to_file(out_shp)

    return out_gdf


def compute_heads(
    date: Timestamp, h1: float, h0: float, interpolated_points: GeoDataFrame
) -> GeoDataFrame:
    """
    Interpolates head values for a given set of points and returns a GeoDataFrame.

    Parameters
    ----------
    date : pandas.Timestamp
        The date for which the head values are computed.
    h1 : float
        The head value at the northernmost point.
    h0 : float
        The head value at the southernmost point.
    interpolated_points : geopandas.GeoDataFrame
        A GeoDataFrame containing the points to be interpolated.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the interpolated points with the `head` and
        `date` columns added.
    """

    interpolated_points["y"] = interpolated_points.geometry.y
    sorted_interpolated_points = interpolated_points.sort_values(
        by="y", ascending=False
    ).reset_index(drop=1)

    out_gdf = sorted_interpolated_points.copy()
    _ = np.zeros_like(out_gdf.index)
    # interpolate
    heads = np.linspace(h1, h0, len(_))  # north comes before
    out_gdf["head"] = heads
    out_gdf["date"] = date
    out_gdf = out_gdf[["date", "head", "geometry"]]

    return out_gdf


def prepare_table(csv_path: Path):
    """
    Reads a CSV file and returns a sorted and pivoted DataFrame.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the sorted and pivoted data from the CSV file.
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df.Date)
    sorted_df = df.sort_values(by=["UTM33U_HW"], ascending=False).reset_index(drop=True)
    cols_north_order = ["Date"] + list(sorted_df.Name_Station.unique())
    pivot_df = (
        pd.pivot_table(
            sorted_df, values="River_WL", columns="Name_Station", index=["Date"]
        )
        .rename_axis(columns="")
        .reset_index()
    )

    pivot_north_order_df = pivot_df[cols_north_order]

    return pivot_north_order_df


def heads_from_table(pivot_table, stations_group, interpolated_points):
    """
    Interpolates head values for a given set of stations and returns a DataFrame.

    Parameters
    ----------
    pivot_table : pandas.DataFrame
        A DataFrame containing the water level data for each station.
    stations_group : list
        A list of station names in the order of the northernmost to the southernmost.
    interpolated_points : geopandas.GeoDataFrame
        A GeoDataFrame containing the points to be interpolated.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the interpolated head values for the given stations and points.
    """
    stations = pivot_table.columns[1:]
    drop_stations = [station for station in stations if station not in stations_group]
    pivot_slice = pivot_table.drop(drop_stations, axis=1)

    # order = date, h1, h0
    heads_df = pivot_slice.apply(
        lambda row: compute_heads(*row, interpolated_points=interpolated_points), axis=1
    )

    return heads_df


# vectorization
def contatenate_parts(df1, df2):
    """concat two dfs and reset index
    to be vectorized through numpy
    """
    return pd.concat([df1, df2]).reset_index(drop=True)


vectorized_concat = np.vectorize(contatenate_parts)


def export_results(
    edit_path: Path, df1: DataFrame, df2: DataFrame, drop_date_column=False, save=False
):
    """
    Concatenates two DataFrames and saves the interpolated head values to shapefiles.

    Parameters
    ----------
    edit_path : pathlib.Path
        The path to the directory where the shapefiles will be saved.
    df1 : pandas.DataFrame
        A DataFrame containing the water level data for the northernmost stations.
    df2 : pandas.DataFrame
        A DataFrame containing the water level data for the southernmost stations.
    drop_date_column : bool, optional
        If True, drops the `date` column from the saved shapefiles. Default is False.
    save : bool, optional
        If True, saves the interpolated head values to shapefiles. Default is False.

    Returns
    -------
    list of pandas.DataFrame
        A list of DataFrames containing the interpolated head values.
    """
    heads_arr = vectorized_concat(df1, df2)

    if save:
        for df in heads_arr:
            fn = str(df.date.unique()[0].date()) + "_heads.shp"
            save_path = edit_path.joinpath(fn)
            if drop_date_column:
                out_df = df.drop("date", axis=1)
            else:
                out_df = df.copy()
                out_df["date"] = df.date.astype("str")
            out_df.to_file(save_path)

    return heads_arr


def plot_results(heads_arr: np.ndarray) -> matplotlib.figure.Figure:
    """
    Plots the interpolated head values for a list of DataFrames.

    Parameters
    ----------
    heads_arr : list of pandas.DataFrame
        A list of DataFrames containing the interpolated head values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    num_plots = len(heads_arr)
    min_head = np.min(list(map(lambda x: x["head"].min(), heads_arr)))
    max_head = np.min(list(map(lambda x: x["head"].max(), heads_arr)))
    num_cols = 6
    num_rows = (num_plots + 1) // num_cols + 1  # + 1 for legend

    gridspec_kw = {"height_ratios": [5, 5, 1]}

    fig = plt.figure(figsize=(20, 10))
    gs = matplotlib.gridspec.GridSpec(num_rows, num_cols, height_ratios=[5, 5, 5])

    for i, df in enumerate(heads_arr):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(gs[row, col])
        df.plot(column="head", legend=False, ax=ax, vmin=min_head, vmax=max_head)
        date = str(df.date.unique()[0].date())
        ax.set_title(f"Heads for {date}")
        [tick.set_rotation(45) for tick in ax.get_xticklabels()]

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=2)

    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    fig.subplots_adjust(bottom=0.05, wspace=0.4, hspace=0.8)

    # create colorbar
    cmap = matplotlib.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=min_head, vmax=max_head)
    # Create a new axis for the colorbar
    cax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
    # Create the colorbar
    cb = matplotlib.colorbar.ColorbarBase(
        cax, cmap=cmap, norm=norm, orientation="horizontal"
    )

    plt.tight_layout()

    plt.show()
    return fig
