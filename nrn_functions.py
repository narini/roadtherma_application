# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:11:20 2024

@author: NRN
"""
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
from plotting import _calculate_tolerance_vs_percentage_high_gradient

def heatmaps_temperature_pixels(ax, pixels, distance, pixel_width, include_colorbar=True):
    """
    Funktion til at plotte heatmaps i toppen af streamlit koden. 
    Kopier af original road therma funktioner. 
    
    Make a heatmap of the temperature columns in the dataframe.
    
    ax: den aksel på et subplot figuren tilgår 
    pixels: den dataframe der skal vises på heatmappet. Indeholder temperatur værdier der bruges til
        farvekoden og vejen opløst i pixels
    distance: serie med distance værdierne der passer til pixels dataframen
    pixel_width: angiver bredten af en pixel. 
    """
    mat = ax.imshow(
        pixels,
        aspect="auto",
        cmap='RdYlGn_r',
        extent=(0, pixels.shape[1] * pixel_width, distance.iloc[-1], distance.iloc[0]) #floats (left, right, bottom, top), (0, pixels.shape[1] * pixel_width, distance.iloc[-1], distance.iloc[0])
    )
    if include_colorbar==True:
        plt.colorbar(mat, ax=ax, label='Temperature [C]')
    
def heatmap_identified_road(ax, pixel_category, distance, pixel_width, categories):
    """
    Figur der minder om   plot_cleaning_results(config, metadata, temperatures, pixel_category), men designet til illustrationer
    _categorical_heatmap(ax, pixels, distance, pixel_width, categories)
    
    resultatet kan sættes ind på en af subplots exkser
    
    ax: den aksel på et subplot figuren tilgår 
    pixel_category : dataframe der har de forskellige ting man vil illustrere givet ved kategorier
    distance: serie med distance værdierne der passer til pixels dataframen
    pixel_width: angiver bredten af en pixel. 
    
    """
    colors = ["dimgray", "firebrick", "springgreen"]
    mat = ax.imshow(
        pixel_category,
        aspect='auto',
        vmin=np.min(pixel_category) - .5, 
        vmax=np.max(pixel_category) + .5,
        cmap=ListedColormap(colors[:len(categories)]),
        extent=(0, pixel_category.shape[1] * pixel_width, distance.iloc[-1], distance.iloc[0])
    )
    # tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ax=ax, ticks=np.arange(
        np.min(pixel_category), np.max(pixel_category) + 1))
    cbar.set_ticklabels(categories)
    
#%## plot af kun identificeret vej og ikke med roller på
def create_identified_road_pixels(pixels_raw, trim_result, lane_result, roadwidths):
    """
    som create_trimming_result_pixels, bare kun med vej og ikke vej
    """
    pixel_category = np.zeros(pixels_raw.shape, dtype='int')
    trim_col_start, trim_col_end, trim_row_start, trim_row_end = trim_result
    lane_start, lane_end = lane_result
    view = pixel_category[trim_row_start:trim_row_end, trim_col_start:trim_col_end]

    for longitudinal_idx, (road_start, road_end) in enumerate(roadwidths):
        view[:, lane_start:lane_end][longitudinal_idx, road_start:road_end] = 1

    return pixel_category


def temperature_to_csv( temperatures, metadata, road_pixels):
    """
    Denne funktion retunerer de rå temperatur data kun for de pixels der er vejen. 
    Kombineres med metadata også
    
    Retunere den dataframe der skal gemmes som csv. 
    

    """
    temperatures = temperatures.copy()
    temperatures.values[~road_pixels] = 'NaN'
    df = pd.merge(metadata,temperatures, how='inner', copy=True, left_index=True, right_index=True)
    return df 

def detections_to_csv( temperatures, metadata, road_pixels, detected_pixels):
    """
    Denne funktion retunerer en dataframe med 0 for ikke vej pixel,
    1 for vej pixels og 2 for dedekterede pixels
    
    temperatures: den trimmede temperatur df
    metadata: df med meta data
    road_pixels: df med True der hvor der er vej
    detect_pixels: df med True der hvor der er dedekteret
    """
    data_width = temperatures.values.shape[1]
    temperatures = temperatures.copy()
    temperatures.values[~road_pixels] = 0
    temperatures.values[road_pixels] = 1
    temperatures.values[detected_pixels & road_pixels] = 2
    temperatures[f'percentage'] = 100 * (np.sum(detected_pixels & road_pixels, axis=1) / data_width)
    df = pd.merge(metadata, temperatures , how='inner', copy=True, left_index=True, right_index=True)
    return df 

def temperature_mean_to_csv(temperatures, road_pixels):
    """
    Funktion der retunere dataframe med gennemsnits temperatur for hver distance for vejen. 
    """
    temperatures = temperatures.copy()
    temperatures.values[~road_pixels] = 'NaN'
    temperatures['temperature_sum'] = np.nanmean(temperatures.values, axis=1)
    df = temperatures[['temperature_sum']]
    return df


def plot_statistics_gradientPlot(title, temperatures, roadwidths, road_pixels, tolerance):
    """
    Tager udgangspunkt i plotting.plot_statistics , men dele op så de to plots er hver for sig. 
    Den ene er meget tidskrævende og skal derfor kunne slås fra. Denne del 
    plotter "Percentage high gradient as a function of tolerance" 

    """
    tol_start, tol_end, tol_step = tolerance
    tolerances = np.arange(tol_start, tol_end, tol_step)

    fig_stats, ax1 = plt.subplots()
    ax1.set_title(title)

    # Plot showing the percentage of road that is comprised of high gradient pixels for a given gradient tolerance
    high_gradients = _calculate_tolerance_vs_percentage_high_gradient(
        temperatures, roadwidths, road_pixels, tolerances)
    ax1.set_title('Percentage high gradient as a function of tolerance')
    ax1.set_xlabel('Threshold temperature difference [C]')
    ax1.set_ylabel('Percentage of road whith high gradient.')
    sns.lineplot(x=tolerances, y=high_gradients, ax=ax1)

    return fig_stats

def plot_statistics_TempDistributionPlot(title, temperatures, road_pixels, limits):
    """
    Tager udgangspunkt i plotting.plot_statistics , men dele op så de to plots er hver for sig. 
    Den ene er meget tidskrævende og skal derfor kunne slås fra. 
    Denne del plotter "Road temperature distribution"
    
    limits = [x_lower, x_higher]

    """

    fig_stats, ax1 = plt.subplots()
    ax1.set_title(title)

    # Plot showing histogram of road temperature
    ax1.set_title('Road temperature distribution')
    ax1.set_xlabel('Temperature [C]')
    distplot_data = temperatures.values[road_pixels]
    sns.histplot(distplot_data, color="m", ax=ax1,
                 stat='density', discrete=True)#, kde=True)
    ax1.set_xlim(limits)
    return fig_stats