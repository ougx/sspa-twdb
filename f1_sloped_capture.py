# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 07:26:36 2021
update 1:
    Transverse Mercator projection is used at each well locally. After plume calculation, the projection is converted back to latlon.

@author: Michael Ou
"""

import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon
import geopandas as gpd
import a0
from pyproj import Proj
from shapely.ops import transform
from shapely.affinity import rotate
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline

#%% functions

def xbar(q0, B, Q, x):
    return (2 * np.pi * q0 * B * x) / Q

def tbar(q0, B, Q, t, n):
    return (2 * np.pi * q0**2 * B * t) / n / Q

def solved_xbar_max(tbar, x0=0, ybar=0, tol=1e-10):
    if pd.isna(tbar) or tbar == 0:
        return tbar
    if ybar == 0.:
        xnew = np.log(1 + x0) + tbar
    else:
        xnew = np.log(np.cos(ybar) + x0 * np.sin(ybar) / ybar) + tbar

    if np.abs(xnew - x0) < tol:
        return xnew
    else:
        return solved_xbar_max(tbar, x0=xnew, ybar=ybar, tol=tol)



def solved_xbar_min(tbar, x0=0, ybar=0, tol=1e-10):
    if pd.isna(tbar) or tbar == 0:
        return tbar

    if ybar == 0.:
        xnew = np.exp(x0 - tbar) - 1
    else:
        xnew = (np.exp(x0 - tbar) - np.cos(ybar)) * ybar / np.sin(ybar)

    if np.abs(xnew - x0) < tol:
        return xnew
    else:
        return solved_xbar_min(tbar, x0=xnew, ybar=ybar, tol=tol)


def solved_ybar(tbar, xbar=0, y0=0.5, tol=1e-10):
    # using the Newton's method
    if pd.isna(tbar) or tbar == 0:
        return tbar
    elif xbar == 0:
        return np.arccos(np.exp(-tbar))#, 0
    else:
        f = lambda y: np.cos(y) + xbar * np.sin(y) / y - np.exp(xbar - tbar)
        f_ = lambda y: -np.sin(y) - xbar * np.sin(y) / (y**2) + xbar * np.cos(y) / y

        # f = lambda y: np.log(np.cos(y) + xbar * np.sin(y) / y) - (xbar - tbar)
        # f_ = lambda y: (-np.sin(y) - xbar * np.sin(y) / (y**2) + xbar * np.cos(y) / y) / (np.cos(y) + xbar * np.sin(y) / y) - np.exp(xbar - tbar)

        solver = root_scalar(f, bracket=(tol*0.1, np.pi*(1-tol)), x0=0.5, fprime=f_, xtol=tol, method='brenth')
        if solver.converged:
            return solver.root
        else:
            solver = root_scalar(f, bracket=(tol*0.1, np.pi*(1-tol)), x0=0.5, fprime=f_, xtol=tol, method='bisect')
            if solver.converged:
                return solver.root
            else:
                raise ValueError(f'solved_ybar fails: tbar = {tbar}; xbar = {xbar}')
                
def createPlumePoly(t, q0, B, n, Q, x=0, y=0, ratation=0):

    coeff = 2 * np.pi * q0 * B / Q
    tbar = coeff * q0 * t / n
    x0 = x
    y0 = y

    if np.isnan(tbar):
        return Point(x0, y0).buffer(1e-10)

    # calculate the xbar range
    xbar_min = solved_xbar_min(tbar)
    xbar_max = solved_xbar_max(tbar)

    # left
    left = np.array([[x, solved_ybar(tbar, x)] for x in np.linspace(xbar_min, 0, 10)[1:]])
    left = np.concatenate([left[::-1], [[xbar_min, 0]], left * [1, -1]])

    cs = CubicSpline(left[::-1,1], left[::-1,0])
    yyl = np.linspace(left[0,1], left[-1,1], 101)
    xxl = cs(yyl)
    # plt.plot(left[:,1], left[:,0]);plt.plot(yyl, xxl)

    # middle
    ymiddle = max(xbar_max - 2, 2*xbar_max/3)
    middle = np.array([[x, solved_ybar(tbar, x)] for x in np.linspace(0, ymiddle, 10)][1:])
    middle = np.concatenate([left[:1], middle])

    cs = CubicSpline(middle[:,0], middle[:,1])
    xxm = np.linspace(middle[0,0], middle[-1,0], 50)[1:]
    yym = cs(xxm)

    # right
    right = np.array([[x, solved_ybar(tbar, x)] for x in np.linspace(ymiddle, xbar_max, 10)[1:-1]])
    right = np.concatenate([middle[-1:], right])
    right = np.concatenate([right, [[xbar_max, 0]], right[::-1] * [1,-1]])

    cs = CubicSpline(right[::-1,1], right[::-1,0])
    yyr = np.linspace(right[0,1], right[-1,1], 101)[1:-1]
    xxr = cs(yyr)
    # plt.plot(right[:,1], right[:,0]);plt.plot(yyr, xxr)

    xx = np.concatenate([xxl, xxm, xxr, xxm[::-1]]) / coeff + x0
    yy = np.concatenate([-yyl, yym, yyr, -yym[::-1]]) / coeff + y0

    # plt.plot(xx, yy)

    poly = Polygon(np.stack([xx,yy]).T)
    return rotate(poly, ratation, (x0, y0))


#%%
def create_shapefile(
        csv_in=a0.proj_path('04_SourceCode', 'Data', 'Nacatoch', 'Output_Details.csv'),
        shp_out=None):

    df_input = pd.read_csv(csv_in)
    
    # convert it to shapefile
    df_input = gpd.GeoDataFrame(df_input, geometry=df_input.apply(lambda x: Point(x.Longitude, x.Latitude), axis=1))
    df_input.crs = 'epsg:4269' # latlon
    
    # df_input = df_input.to_crs('+proj=utm +zone=14 +datum=NAD83 +units=us-ft')
    # df_input.to_file(a0.proj_path('04_SourceCode', 'Data', 'Nacatoch', 'df_input_Projected.shp'))

    df_input['coeff'] = 2 * np.pi * df_input.DarcyFlow * df_input.B / df_input.InjectionRate
    df_input['tbar'] = df_input['coeff'] * df_input.DarcyFlow * df_input.ElapsedTime / df_input.Porosity

    df_input['xmin'] = df_input['tbar'].apply(solved_xbar_min) / df_input['coeff']
    df_input['xmax'] = df_input['tbar'].apply(solved_xbar_max) / df_input['coeff']

    # p = Proj('+proj=utm +zone=14 +datum=NAD83 +units=us-ft +no_defs', preserve_units=True)
    # df_input['rotation'] = df_input.apply(lambda x: np.random.randint(-180, 180), axis=1)
    # df_input['rotation'] = np.random.randint(-180, 180)
    df_input['rotation'] = 45

    gg = []
    for i, r in df_input.iterrows():
        # print(i, r.ElapsedTime, r.InjectionRate)
        # i = 0; r = df_input.loc[i]
        centroid = r.geometry.centroid
        proj = Proj(f'+proj=tmerc +lat_0={centroid.y} +lon_0={centroid.x} +units=us-ft')
        g = createPlumePoly(
            t = r.ElapsedTime,
            q0 = r.DarcyFlow,
            B = r.B,
            n = r.Porosity,
            Q = r.InjectionRate,
            x = 0.,
            y = 0.,
            ratation = r.rotation
        )
        gg.append(transform(lambda x, y: proj(x,y, inverse=True), g, ))
        # gg.append(g)

    output = df_input.copy()
    output.geometry = gg
    shp_out = shp_out or 'output.shp'
    output.to_file(shp_out)

    print(f'Output written to {shp_out}')
