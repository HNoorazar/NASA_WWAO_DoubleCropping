# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="rNWduYF8ELxV" outputId="9bdfaaa7-7b02-401e-dd0f-4095d69f74c8"
# !pip install shutup
import shutup
shutup.please() # kill some of the messages

# %% id="vpd1DlamoKpN"
import pickle

# %% [markdown] id="1k8_CIo5DqvC"
# # Print Local Time
#
# colab runs on cloud. So, the time is not our local time.
# This page is useful to determine how to do this.

# %% colab={"base_uri": "https://localhost:8080/"} id="zxAcDFMxDwIm" outputId="6707570d-a33d-406b-f58b-91d517792c7e"
# !rm /etc/localtime
# !ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime
# !date

# %% [markdown] id="Nyg94VHsEK8B"
# # **geopandas and geemap must be installed every time!!!**
#

# %% id="F9b71mZbEZQj" colab={"base_uri": "https://localhost:8080/"} outputId="852f555c-4544-4f19-b0c0-b8022fcb8b45"
# # !pip install geopandas geemap
# Installs geemap package
import subprocess

try:
    import geemap
except ImportError:
    print('geemap not installed. Must be installed every tim to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

    print('geopandas not installed. Must be installed every time to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geopandas'])
    subprocess.check_call(["python", '-m', 'pip', 'install', 'google.colab'])

# %% [markdown] id="StQCEezUEajV"
# # **Authenticate and import libraries**
#
# We have to impor tthe libraries we need. Moreover, we need to Authenticate every single time!

# %% id="vWONKBNXEpCD" colab={"base_uri": "https://localhost:8080/"} outputId="ff50e712-4234-4eb1-eaaf-9e459c86958e"
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
import json, geemap, ee

import scipy # we need this for savitzky-golay

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time, datetime
from datetime import date


try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# %% [markdown] id="Magk2Zr_ExC1"
# ### **Mount Google Drive and import my Python modules**
#
# Here we are importing the Python functions that are written by me and are needed; ```NASA core``` and ```NASA plot core```.
#
# **Note:** These are on Google Drive now. Perhaps we can import them from GitHub.

# %% colab={"base_uri": "https://localhost:8080/"} id="IxshCK3dE8lf" outputId="09ed1b7f-0593-480f-b885-c4f5e19754b3"
# Mount YOUR google drive in Colab
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0,"/content/drive/My Drive/Colab Notebooks/")
import NASA_core as nc
import NASA_plot_core as ncp
import GEE_Python_core as gpc

# **Change Current directory to the Colab folder on Google Drive**
import os
os.chdir("/content/drive/My Drive/Colab Notebooks/") # Colab Notebooks
# # !ls

# %% [markdown] id="FIrLJ4-pFQEz"
# # Please tell me where to look for the shapefile!

# %% id="juqgx2EpFUU1" colab={"base_uri": "https://localhost:8080/", "height": 130} outputId="2507825b-3e67-4953-d7a5-ef15c03445e9"
# shp_path = "/Users/hn/Documents/01_research_data/NASA/shapefiles/Grant2017/Grant2017.shp"
# shp_path = "/content/My Drive/NASA_trends/shapefiles/Grant2017/Grant2017.shp"
shp_path = "/content/drive/MyDrive/NASA_trends/shapefiles/Grant2017/Grant2017.shp"
shp_path = "/content/drive/MyDrive/NASA_trends/shapefiles/" + \
            "Grant_4Fields_poly_wCentroids/Grant_4Fields_poly_wCentroids.shp"

# we read our shapefile in to a geopandas data frame using the geopandas.read_file method
# we'll make sure it's initiated in the EPSG 4326 CRS
SF = gpd.read_file(shp_path, crs='EPSG:4326')

### for possible future use grab the data part of the shapefile
SF_data = SF[["ID", "Acres", "county", "CropTyp", \
              "DataSrc", "Irrigtn", "LstSrvD"]].copy()
SF_data.drop_duplicates(inplace=True)
print (SF_data.shape)

"""
   Drop extra useless columns. Saves space.**
   Also, GEE behaves strangely. It has problem with Notes column before
"""
# SF = SF.drop(columns=["Notes", "TRS", "IntlSrD", "RtCrpTy", "Shp_Lng", "Shap_Ar","CropGrp","CropTyp","Acres","Irrigtn","LstSrvD","DataSrc","county","ExctAcr","cntrd_ln","cntrd_lt"])
SF = SF.drop(columns=["CropTyp","Acres","Irrigtn",
                      "LstSrvD","DataSrc","county","ExctAcr",
                      "cntrd_ln","cntrd_lt"])

SF.head(2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 183} id="EzT2oOS5Fi61" outputId="0e0348b4-dfd4-4f5f-8c13-1d045c66a8d0"
long_eq = "=============================================================================="
print (type(SF))
print (long_eq)
print (f"{SF.shape=}", )
print (long_eq)
SF.head(2)

# %% id="H18AR2kV9zSy"
IDs = list(SF_data.ID.unique())

# %% [markdown] id="JVOsJFXkFoCH"
# # **Form Geographical Regions**
#
#   - First, define a big region that covers Eastern Washington.
#   - Convert shapefile to ```ee.featurecollection.FeatureCollection```.
#

# %% id="Ihy6wVxAHe_A"
xmin = -125.0;
ymin = 45.0;
xmax = -116.0;
ymax = 49.0;

xmed = (xmin + xmax) / 2.0;
ymed = (ymin+ymax) / 2.0;

WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed, ymax], [xmed, ymin], [xmin, ymin]]);
WA2 = ee.Geometry.Polygon([[xmed, ymin], [xmed, ymax], [xmax, ymax], [xmax, ymin], [xmed, ymin]]);
WA = [WA1,WA2];
big_rectangle = ee.FeatureCollection(WA);
SF = geemap.geopandas_to_ee(SF)

# %% [markdown] id="MLucJYy7Hgrr"
# ## **WARNING!**
# For some reason the function ```feature2ee(.)``` does not work ***when*** it is imported from ```core``` module. (However, it works when it is directly written here!!!) So, What the happens with the rest of functions, e.g. smoothing functions, we want to use here?

# %% id="S3GBvrufIuZ8"
# was named "banke" in https://bikeshbade.com.np/tutorials/Detail/?title=Geo-pandas%20data%20frame%20to%20GEE%20feature%20collection%20using%20Python&code=13
# Grant_4Fields_poly_wCentroids_EEFC_from_Func = feature2ee(shp_path)
# Grant_4Fields_poly_wCentroids_EEFC_from_Func = gpc.feature2ee(shp_path)

# %% [markdown] id="_WohFsaWI3_w"
# # Visualize the big region encompassing the Eastern Washington

# %% colab={"base_uri": "https://localhost:8080/", "height": 621, "referenced_widgets": ["c7e70fcc940644ae9a47759abe0089fe", "90b9fb8106b64657ab2464d6ab05014d", "6f1ab865db6d442f93b84cf6d1bfdd22", "5d41b53701734f2e950139b5fb2aa084", "06ffecb2955849e7ad5490a3d2637d62", "a04687c0432a4a428f2ba69141ae63c2", "0ec2e960270d47a88293fa1d4d600241", "efae7ceb4d524c3cb385b1a910dd0512", "7376c9d45ab54ceba779b2878d31d2b0", "61d98ca7dc2f41789f8b6f07c9c09f9c", "3a84bfb94d2d470ea627459d062aeebc", "ddc3efa9b4ad47ef8569728a5779af6c", "4311bbb9d67b4d678bc0ad5cff32a0dc", "26c5e5e07f1d49aba73e495c78c81f3b", "bb09571b364c402aa1b5834e954d8fd2", "236be085373c48ba9bbbd9688c046104", "0c12b645b86349adad6f53bf63055a62", "600ac7ab6e3647e6991f854de0d927b0", "c3d1b143a3394f71b84f554015ea02fa", "1eeb5691b18e41dd8b4419269f53751a", "a53e2a5d9b624ae7b5947b5ba2e4937a", "bba0f1ebc08c4244857aa91f351a172b", "94476aae672944379a62d6bd4d2ce4b6", "8051a98f2f8c45efa912fe9dd39df7a9", "47c5038abab64a36b5db4c3cedd099b4", "4c0781f50960483fa31d9ee974f1430d", "eefe308b4a7441d2b53e743843f7bcaf", "add3205afb754a00a18e4428351c1e60"]} id="n_VdDpeqI66Q" outputId="22cdd1df-0093-4b84-cd03-3bfca191699f"
Map = geemap.Map(center=[ymed, xmed], zoom=7)
Map.addLayer(WA1, {'color': 'red'}, 'Western Half')
Map.addLayer(WA2, {'color': 'red'}, 'Eastern Half')
Map.addLayer(SF, {'color': 'blue'}, 'Fields')
Map

# %% [markdown] id="OuKWgSEXJD7B"
# # Define Parameters

# %% id="Og1nGEviJJ1_"
start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD
cloud_perc = 70

# %% [markdown] id="rqJXjNo1JWmR"
# ## Fetch data from GEE

# %% colab={"base_uri": "https://localhost:8080/"} id="4fOcXB4cJPcC" outputId="231c61f4-58a6-4198-88e6-d187cceebed2"
# %%time
imageC = gpc.extract_sentinel_IC(big_rectangle, start_date, end_date, cloud_perc);
print ("The size of [imageC.size().getInfo()] is [{:.0f}].".format(imageC.size().getInfo()))

reduced = gpc.mosaic_and_reduce_IC_mean(imageC, SF, start_date, end_date)

needed_columns = ["ID", "EVI", 'NDVI', "system_start_time"]
reduced = geemap.ee_to_pandas(reduced, selectors=needed_columns)
reduced = reduced[needed_columns]

# %% [markdown] id="3g13uBmAJaE1"
# # Export output to Google Drive only
#
# We advise you to do it. If Python/CoLab kernel dies, then,
# previous steps should not be repeated.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="JktTEw_-JfK4" outputId="575bbec3-dae4-4a64-a394-b270eb6226f1"
# %%time
export_raw_data = True

# if export_raw_data==True:
#     outfile_name = "Grant_4Fields_poly_wCentroids_colab_output"
#     task = ee.batch.Export.table.toDrive(**{
#                                         'collection': reduced,
#                                         'description': outfile_name,
#                                         'folder': "colab_outputs",
#                                         'selectors':["ID", "Acres", "county", "CropTyp", "DataSrc", \
#                                                      "Irrigtn", "LstSrvD", "EVI", 'NDVI', "system_start_time"],
#                                         'fileFormat': 'CSV'})
#     task.start()

#     import time
#     while task.active():
#         print('Polling for task (id: {}). Still breathing'.format(task.id))
#         time.sleep(59)


if export_raw_data == True:
    outfile_name = "Perry_WSDA_Demo_" + start_date + "_" + end_date + ".csv"
    reduced.to_csv('/content/drive/MyDrive/colab_outputs/'+ outfile_name)

# %% [markdown] id="JWEL1ff9JzI-"
# # **Smooth the data**
#
# This is the end of Earh Engine Part. Below we start smoothing the data and carry on!
#
# First, all these steps can be done behind the scene. But doing them here, one at a time, has the advantage that if something goes wrong in the middle, then
# we do not lose the good stuff that was done earlier!
# For example, of one of the Python libraries/packages needs to be updated in the middle of the way
# we do not have to start doing everything from the beginning!
# <p>&nbsp;</p>
#
# Start with converting the type of ```reduced``` from ```ee.FeatureCollection``` to ```dataframe```.
#
# - For some reason when converting the ```ee.FeatureCollection``` to ```dataframe``` the function has a problem with the ```Notes``` column! So, I remove the unnecessary columns.
#
# **NA removal**
#
# Even though logically and intuitively all the bands should be either available or ```NA```, I have seen before that sometimes ```EVI``` is NA while ```NDVI``` is not. Therefore, I had to choose which VI we want to use so that we can clean the data properly. However, I did not see that here.  when I was testing this code for 4 fields.
#
# Another suprising observation was that the output of Colab had more data compared to its JS counterpart!!!
#
# # **Define the VI parameter we want to work with**

# %% id="QzKOC2DJ5Ulh"
VI_idx = "NDVI"

# %% id="sSDOXhv0SKNb"
# reduced = reduced[reduced["system_start_time"].notna()]
reduced = reduced[reduced[VI_idx].notna()]
reduced.reset_index(drop=True, inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="TrlVRbS8SPe-" outputId="535c9ef4-436e-47b1-8b10-96754e0e70d4"
# Add human readable time to the dataframe
reduced = nc.add_human_start_time_by_system_start_time(reduced)
reduced.head(2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="sqXm2ZKkSedK" outputId="5a5632e8-d452-45b8-d2b5-f3cda9b8dd47"
#  Pick a field
a_field = reduced[reduced.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(axis='y', which="both")

ax.scatter(a_field['human_system_start_time'], a_field[VI_idx], s=40, c='#d62728');
ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
        linestyle='-',  linewidth=3.5, color="#d62728", alpha=0.8,
        label=f"raw {VI_idx}")
plt.ylim([-0.5, 1.2]);
ax.legend(loc="lower right");
# ax.set_title(a_field.CropTyp.unique()[0]);

# %% [markdown] id="zTYd9dqLSsxn"
# # Efficiency
#
# Can we make this more efficient by doing the calculations in place as opposed to creating a new ```dataframe``` and copying stuff. Perhaps ```.map(.)``` too.
#
# **Remove outliers**

# %% id="bAHLlnE7Szjw"
reduced["ID"] = reduced["ID"].astype(str)
p = np.sort(reduced["ID"].unique())


# %% colab={"base_uri": "https://localhost:8080/"} id="FAxGeCA-TEoF" outputId="33983664-ac38-4f5a-a8f4-ba65353e1618"
no_outlier_df = pd.DataFrame(data = None,
                         index = np.arange(reduced.shape[0]),
                         columns = reduced.columns)
counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = reduced[reduced["ID"]==a_poly].copy()
    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        ##************************************************
        #
        #    Set negative index values to zero.
        #
        ##************************************************
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = VI_idx)
        no_Outlier_TS.loc[no_Outlier_TS[VI_idx
                                        ] < 0 , VI_idx] = 0

        """
        it is possible that for a field we only have x=2 data points
        where all the EVI/NDVI is outlier. Then, there is nothing to
        use for interpolation. So, hopefully interpolate_outliers_EVI_NDVI is returning an empty data table.
        """
        if len(no_Outlier_TS) > 0:
            no_outlier_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]

# Sanity check. Will neved occur. At least should not!
no_outlier_df.drop_duplicates(inplace=True)

# %% [markdown] id="L0Bu6xyNTIN7"
# **Remove the jumps**
#
# Maybe we can remove old/previous dataframes to free memory up!

# %% colab={"base_uri": "https://localhost:8080/"} id="bn2-bHZ4Tfk9" outputId="770648c8-6052-4e93-8158-2d19d948d5fe"
noJump_df = pd.DataFrame(data = None,
                         index = np.arange(no_outlier_df.shape[0]),
                         columns = no_outlier_df.columns)
counter, row_pointer = 0, 0

for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = no_outlier_df[no_outlier_df["ID"]==a_poly].copy()

    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    ################################################################

    no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie = curr_field,
                                                        give_col = VI_idx
                                                        ,
                                                        maxjump_perDay = 0.018)

    noJump_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

noJump_df['human_system_start_time'] = pd.to_datetime(noJump_df['human_system_start_time'])

# Sanity check. Will neved occur. At least should not!
print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape))
noJump_df.drop_duplicates(inplace=True)
print ("Shape of noJump_df after dropping duplicates is {}.".format(noJump_df.shape))

del(no_Outlier_TS)

# %% [markdown] id="evXDiJKxTjEQ"
# **Regularize**
#
# Here we regularize the data. "Regularization" means pick a value for every 10-days. Doing this ensures
#
# 1.   all inputs have the same length,
# 2.   by picking maximum value of a VI we are reducing the noise in the time-series by eliminating noisy data points. For example, snow or shaddow can lead to understimating the true VI.
#
# Moreover, here, I am keeping only 3 columns. As long as we have ```ID``` we can
# merge the big dataframe with the final result later, here or externally.
# This will reduce amount of memory needed. Perhaps I should do this
# right the beginning.

# %% colab={"base_uri": "https://localhost:8080/"} id="fynzOVKCT3qE" outputId="b1a81196-ecfe-4136-e952-6cc968253cf6"
# %%time

# define parameters
regular_window_size = 10
reg_cols = ['ID', 'human_system_start_time', VI_idx] # system_start_time list(noJump_df.columns)

st_yr = noJump_df.human_system_start_time.dt.year.min()
end_yr = noJump_df.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size)) # no_days // regular_window_size

nrows = no_steps * len(IDs)
print('st_yr is {}.'.format(st_yr))
print('end_yr is {}.'.format(end_yr))
print('nrows is {}.'.format(nrows))
print (long_eq)


regular_df = pd.DataFrame(data = None,
                         index = np.arange(nrows),
                         columns = reg_cols)
counter, row_pointer = 0, 0

for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = noJump_df[noJump_df["ID"]==a_poly].copy()
    ################################################################
    # Sort by date (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    ################################################################
    regularized_TS = nc.regularize_a_field(a_df = curr_field, \
                                           V_idks = VI_idx, \
                                           interval_size = regular_window_size,\
                                           start_year = st_yr, \
                                           end_year = end_yr)

    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = VI_idx)
    # if (counter == 0):
    #     print ("regular_df columns:",     regular_df.columns)
    #     print ("regularized_TS.columns", regularized_TS.columns)

    ################################################################
    # row_pointer = no_steps * counter

    """
       The reason for the following line is that we assume all years are 366 days!
       so, the actual thing might be smaller!
    """
    # why this should not work?: It may leave some empty rows in regular_df
    # but we drop them at the end.
    regular_df[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
    row_pointer += regularized_TS.shape[0]
    counter += 1

regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
regular_df.drop_duplicates(inplace=True)
regular_df.dropna(inplace=True)

# Sanity Check
regular_df.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
regular_df.reset_index(drop=True, inplace=True)

del(noJump_df)

# %% id="lqzNrGrET74H"

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="ykCINQmQU9xd" outputId="e3413fab-091a-4a7b-c8e3-56c9cdfe2f49"
#  Pick a field
a_field = regular_df[regular_df.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3), sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'],
        a_field[VI_idx],
        linestyle='-', label=VI_idx, linewidth=3.5, color="dodgerblue", alpha=0.8)

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% [markdown] id="IjmgbKcDVTiw"
# **Savitzky-Golay Smoothing**

# %% colab={"base_uri": "https://localhost:8080/"} id="oVhXa8DwVegh" outputId="513c7a3d-4de2-4c2b-ab75-df5e8c1b0179"
# %%time
counter = 0
window_len, polynomial_order = 7, 3

for a_poly in IDs:
    if (counter % 300 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = regular_df[regular_df["ID"]==a_poly].copy()

    # Smoothen by Savitzky-Golay
    SG = scipy.signal.savgol_filter(curr_field[VI_idx].values, window_length=window_len, polyorder=polynomial_order)
    SG[SG > 1 ] = 1 # SG might violate the boundaries. clip them:
    SG[SG < -1 ] = -1
    regular_df.loc[curr_field.index, VI_idx] = SG
    counter += 1

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="7MwDu6q2Vpji" outputId="c4ef40a5-8a9f-41da-fa53-98b024e3db9a"
# Pick a field
an_ID = IDs[0]
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3), sharex='col', sharey='row', gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8, label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="SpyopmVQWiFD" outputId="c85e3ca8-c519-4b1a-96d4-f76ec401d0d1"
regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
# regular_df = pd.merge(regular_df, SF_data, on=['ID'], how='left') # we can do this later.
regular_df.reset_index(drop=True, inplace=True)
regular_df = nc.initial_clean(df=regular_df, column_to_be_cleaned = VI_idx
                              )
regular_df.head(2)

# %% [markdown] id="RCVxZ2CJtUkR"
# **Widen the data to use with ML (other than DL)**

# %% id="GwjpZeyCtdHB"
model = "DL"

if model != "DL":
    VI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37)]
    columnNames = ["ID", "year"] + VI_colnames

    years = regular_df.human_system_start_time.dt.year.unique()
    no_rows = len(IDs) * len(years)

    data_wide = pd.DataFrame(columns=columnNames, index=range(no_rows))
    data_wide.ID = list(IDs) * len(years)
    data_wide.sort_values(by=["ID"], inplace=True)
    data_wide.reset_index(drop=True, inplace=True)
    data_wide.year = list(years) * len(IDs)

    for an_ID in IDs:
        curr_field = regular_df[regular_df.ID == an_ID]
        curr_years = curr_field.human_system_start_time.dt.year.unique()
        for a_year in curr_years:
            curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]
            data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index
            if VI_idx == "EVI":
                data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_field_year.EVI.values[:36]
            elif VI_idx == "NDVI":
                data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = curr_field_year.NDVI.values[:36]

# %% [markdown] id="AYMI0RwcWjc7"
# # Please tell me where to look for the trained models and I will make you happy!

# %% id="4HwojA6Ppui5"
model_dir = "/content/drive/MyDrive/NASA_trends/Models_Oct17/"
model_dir = "/content/drive/MyDrive/Colab Notebooks/Models_Oct17/"

# %% id="cHr4PiBJwfQU"

# %% id="mb3UzKm5jV4S"
import pickle

if model == "KNN":
    winnerModel = "KNN_SG_NDVI_Oct17_AccScoring_Oversample_SR3.sav"
elif model == "SVM":
    winnerModel = "SG_NDVI_SVM_NoneWeight_00_Oct17_AccScoring_Oversample_SR3.sav"
elif model =="RF":
    winnerModel = "SG_NDVI_RF_grid_2_Oct17_AccScoring_Oversample_SR5.sav"
else:
    winnerModel = "01_TL_NDVI_SG_train80_Oct17_oversample5.h5"

###
### Load model and predict
###
if winnerModel.endswith(".sav"):
    ML_model = pickle.load(open(model_dir + winnerModel, "rb"))
    predictions = ML_model.predict(data_wide.iloc[:, 2:])
    pred_colName = model + "_" + VI_idx  + "_preds"
    A = pd.DataFrame(columns=["ID", "year", pred_colName])
    A.ID = data_wide.ID.values
    A.year = data_wide.year.values
    A[pred_colName] = predictions
    predictions = A.copy()
    del A
else:
    from tensorflow.keras.utils import to_categorical, load_img, img_to_array
    from keras.models import Sequential, Model, load_model
    from keras.applications.vgg16 import VGG16
    import tensorflow as tf

    # from keras.optimizers import SGD
    from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.optimizers import SGD
    from keras.preprocessing.image import ImageDataGenerator


    #### Form predictions' dataframe
    predictions = pd.DataFrame({"ID": list(regular_df.ID.unique())})
    predictions["prob_single"] = -1.0

    ML_model = load_model(model_dir + winnerModel) # load model

    image_dir = '/content/drive/MyDrive/colab_outputs/'
    image_name = image_dir + "fly_test.jpg"
    for an_ID in regular_df.ID.unique():
        crr_fld = regular_df[regular_df.ID == an_ID]
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 2.5)
        ax.grid(False)
        ax.plot(
            crr_fld["human_system_start_time"], crr_fld[VI_idx], c="dodgerblue", linewidth=5
        )
        ax.axis("off")
        left = crr_fld["human_system_start_time"].values[0]
        right = crr_fld["human_system_start_time"].values[-1]
        ax.set_xlim([left, right]);
        # the following line also works
        ax.set_ylim([-0.005, 1]);

        plt.savefig(fname = image_name, dpi = 200, bbox_inches = "tight", facecolor = "w")
        plt.close("all")

        img = nc.load_image(image_name)
        predictions.loc[predictions.ID == an_ID, "prob_single"] = ML_model.predict(img, verbose=False)[0][0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="ovFTtxfolhhk" outputId="c772c3cb-8180-4315-b626-4f563972e130"
predictions = pd.merge(predictions, SF_data, on=['ID'], how='left')
predictions

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="QKBAJpkLwJLj" outputId="6bc78695-147c-4472-df4b-fcee53b117d9"
predictions = pd.merge(predictions, SF_data, on=['ID'], how='left')
predictions

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="k4JyjzKv43zL" outputId="d4d1d895-5f97-4d24-e00a-08b2c805c740"
#  Pick a field
an_ID = "102309_WSDA_SF_2017"
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], a_field[VI_idx],
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
        label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");
# label_ = list(predictions.loc[predictions.ID==an_ID, "SVM_NDVI_preds"])[0]
# ax.set_title(f"SVM prediction is {label_}.")
ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% id="VKskUUEQ47Oz" colab={"base_uri": "https://localhost:8080/"} outputId="7f93e3fb-bab2-44d6-fd29-07986382d4e7"

# %% id="W9a6KElI9BZp"
