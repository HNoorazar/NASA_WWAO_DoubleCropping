import numpy as np
import pandas as pd
import geopandas as gpd
import os, os.path
import ee

###
### These will be more generalized functions of remote_sensing_core.py
### Hence, less hard coding, which implies column/variavle wise we
### will be minimalistic. e.g. column: lastSurveydate should not be included
### here.
###

###########################################################


##################################################
###
###         NDVI - Landsat-8
###


def addNDVI_to_image_Landsat8(image):
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    return image.addBands(ndvi)


def add_NDVI_collection_Landsat8(image_IC):
    NDVI_IC = image_IC.map(addNDVI_to_image_Landsat8)
    return NDVI_IC


##################################################
###
###         NDVI - Landsat-7, 5, and 4
###


def addNDVI_to_image_Landsat754(image):
    ndvi = image.normalizedDifference(["SR_B4", "SR_B3"]).rename("NDVI")
    return image.addBands(ndvi)


def add_NDVI_collection_Landsat754(image_IC):
    NDVI_IC = image_IC.map(addNDVI_to_image_Landsat754)
    return NDVI_IC


##################################################
###
###         EVI - Landsat-7, 5, and 4
###
def addEVI_to_image_Landsat754(image):
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))",
        {"NIR": image.select("SR_B4"), "RED": image.select("SR_B3"), "BLUE": image.select("SR_B1")},
    ).rename("EVI")
    return image.addBands(evi)


def add_EVI_collection_Landsat754(image_IC):
    EVI_IC = image_IC.map(addEVI_to_image_Landsat754)
    return EVI_IC


##################################################
###
###         EVI - Landsat-8
###
def addEVI_to_image_Landsat8(image):
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))",
        {"NIR": image.select("SR_B5"), "RED": image.select("SR_B4"), "BLUE": image.select("SR_B2")},
    ).rename("EVI")
    return image.addBands(evi)


def add_EVI_collection_Landsat8(image_IC):
    EVI_IC = image_IC.map(addEVI_to_image_Landsat8)
    return EVI_IC


def extract_satellite_IC(big_rectangle_featC, start_date, end_date, dataSource):
    """
    We need to check L8 and add it to this.
    We need to check if the bands are identical. Which I think is not.
    dataSource:
        L4: 'LANDSAT/LT04/C02/T1_L2'
             life span is 7/16/1982 to 12/14/1993

        L5: 'LANDSAT/LT05/C02/T1_L2'
            life span is 1-March-1984 to 5-June-2013

        L7: 'LANDSAT/LE07/C02/T1_L2'
            life span is 15-April-1999 to 6-April-2022

        L8: 'LANDSAT/LC08/C02/T1_L2'
            life span is 11-Feb-2013 till now (June 2023)
    """

    def clip_function(image):
        return image.clip(geom)

    geom = big_rectangle_featC.geometry()
    # big_rectangle_featC is a feature collection

    newDict = {"original_polygon_1": geom}
    imageC = (
        ee.ImageCollection(dataSource)
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .map(clip_function)
        .sort("system:time_start", True)
    )

    if dataSource == "LANDSAT/LC08/C02/T1_L2":
        imageC = imageC.map(scale_bands_Landsat8)  # scale the damn bands
        imageC = imageC.map(cloudMaskL578_C2L2)  # toss out cloudy pixels

        imageC = add_NDVI_collection_Landsat8(imageC)
        imageC = add_EVI_collection_Landsat8(imageC)

    elif dataSource in [
        "LANDSAT/LT04/C02/T1_L2",
        "LANDSAT/LT05/C02/T1_L2",
        "LANDSAT/LE07/C02/T1_L2",
    ]:
        imageC = imageC.map(scale_bands_Landsat457)  # scale the damn bands
        imageC = imageC.map(cloudMaskL578_C2L2)  # toss out cloudy pixels

        imageC = add_NDVI_collection_Landsat754(imageC)
        imageC = add_EVI_collection_Landsat754(imageC)

    imageC = add_system_start_time_collection(imageC)

    # add original geometry to each image. We do not need to do this really:
    # imageC = imageC.map(function(im){return(im.set(newDict))});
    imageC = imageC.set({"original_polygon": geom, "WSDA": big_rectangle_featC})

    return imageC


def scale_bands_Landsat457(image):
    NIR = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    red = image.select("SR_B3").multiply(0.0000275).add(-0.2)
    blue = image.select("SR_B1").multiply(0.0000275).add(-0.2)

    return image.addBands(NIR, None, True).addBands(red, None, True).addBands(blue, None, True)


def scale_bands_Landsat8(image):
    NIR = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    blue = image.select("SR_B2").multiply(0.0000275).add(-0.2)

    return image.addBands(NIR, None, True).addBands(red, None, True).addBands(blue, None, True)


def cloudMaskL578_C2L2(image):
    """
    This is modified version of Amin's function cloudMaskL8()
    from the notebook called "Copy_of_pixel_level_featureExtraction.ipynb"
    """
    qa = image.select("QA_PIXEL")  ## substitiu a band FMASK
    cloud1 = qa.bitwiseAnd(1 << 3).eq(0)
    cloud2 = qa.bitwiseAnd(1 << 9).eq(0)
    cloud3 = qa.bitwiseAnd(1 << 4).eq(0)

    # mask2 = image.mask().reduce(ee.Reducer.min());
    # return image.updateMask(cloud1).updateMask(cloud2).updateMask(cloud3).updateMask(mask2).copyProperties(image, ["system:time_start"])

    return (
        image.updateMask(cloud1)
        .updateMask(cloud2)
        .updateMask(cloud3)
        .copyProperties(image, ["system:time_start"])
    )


"""
Above this line is Jun 5, 2023 after JS gives memory error
"""


###########################################################
def mosaic_and_reduce_IC_mean_noWSDA(an_IC, a_feature, start_date, end_date):
    """Return mosaiced and reduced imageCollection. Reduction is mean within a region.
    This function is Python version of my functions in JavaScript from forecast project.
    Here I have deleted the commented lines.

    Arguments
    ---------
    an_Ic : imageCollection
    reduction_geometry : ee.FeatureCollection(SF) where SF is the WSDA shapefile

    Returns
    ---------
    reduced  :  imageCollection
                mosaiced and reduced
    """
    an_IC = ee.ImageCollection(an_IC)
    reduction_geometry = a_feature
    start_date_DateType = ee.Date(start_date)
    end_date_DateType = ee.Date(end_date)
    #######**************************************
    # Difference in days between start and end_date
    diff = end_date_DateType.difference(start_date_DateType, "day")

    def list_the_datesish(day):
        return start_date_DateType.advance(day, "day")

    # Make a list of all dates
    date_range = ee.List.sequence(0, diff.subtract(1)).map(list_the_datesish)

    # Funtion for iteraton over the range of dates
    def day_mosaics(date, newlist):
        # Cast
        date = ee.Date(date)
        newlist = ee.List(newlist)
        # Filter an_IC between date and the next day
        filtered = an_IC.filterDate(date, date.advance(1, "day"))
        image = ee.Image(filtered.mosaic())
        # Make the mosaic
        # Add the mosaic to a list only if the an_IC has images
        return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist))

    # Iterate over the range to make a new list, and then cast the list to an imagecollection
    newcol = ee.ImageCollection(ee.List(date_range.iterate(day_mosaics, ee.List([]))))

    def reduce_regions(image):
        return image.reduceRegions(
            **{"collection": reduction_geometry, "reducer": ee.Reducer.mean(), "scale": 10}
        )

    reduced = newcol.map(reduce_regions).flatten()
    return reduced


def mosaic_and_reduce_IC_mean(an_IC, a_feature, start_date, end_date):
    """Return mosaiced and reduced imageCollection. Reduction is mean within a region.
    This function is Python version of my functions in JavaScript from forecast project.
    Here I have deleted the commented lines.

    Arguments
    ---------
    an_Ic : imageCollection
    reduction_geometry : ee.FeatureCollection(SF) where SF is the WSDA shapefile

    Returns
    ---------
    reduced  :  imageCollection
                mosaiced and reduced
    """
    an_IC = ee.ImageCollection(an_IC)
    reduction_geometry = a_feature
    WSDA = an_IC.get("WSDA")
    start_date_DateType = ee.Date(start_date)
    end_date_DateType = ee.Date(end_date)
    #######**************************************
    # Difference in days between start and end_date
    diff = end_date_DateType.difference(start_date_DateType, "day")

    def list_the_datesish(day):
        return start_date_DateType.advance(day, "day")

    # Make a list of all dates
    date_range = ee.List.sequence(0, diff.subtract(1)).map(list_the_datesish)

    # Funtion for iteraton over the range of dates
    def day_mosaics(date, newlist):
        # Cast
        date = ee.Date(date)
        newlist = ee.List(newlist)
        # Filter an_IC between date and the next day
        filtered = an_IC.filterDate(date, date.advance(1, "day"))
        image = ee.Image(filtered.mosaic())
        # Make the mosaic
        # Add the mosaic to a list only if the an_IC has images
        return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist))

    # Iterate over the range to make a new list, and then cast the list to an imagecollection
    newcol = ee.ImageCollection(ee.List(date_range.iterate(day_mosaics, ee.List([]))))

    def reduce_regions(image):
        return image.reduceRegions(
            **{"collection": reduction_geometry, "reducer": ee.Reducer.mean(), "scale": 10}
        )

    reduced = newcol.map(reduce_regions).flatten()
    reduced = reduced.set({"original_polygon": reduction_geometry, "WSDA": WSDA})
    WSDA = ee.Feature(WSDA)
    WSDA = WSDA.toDictionary()

    def set_d_WSDA(imge):
        return imge.set(WSDA)

    reduced = reduced.map(set_d_WSDA)
    return reduced


def extract_sentinel_IC(a_feature, start_date, end_date, cloud_perc):
    def clip_function(image):
        return image.clip(geom)

    geom = a_feature.geometry()  # a_feature is a FeatureCollection
    # newDict = {"original_polygon_1": geom}
    imageC = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .map(clip_function)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_perc))
        .sort("system:time_start", True)
    )

    imageC = imageC.map(maskS2clouds)  # toss out cloudy pixels
    imageC = add_NDVI_collection_Sentinel(imageC)  # add NDVI as a band
    imageC = add_EVI_collection_Sentiel(imageC)  # add EVI as a band
    imageC = add_system_start_time_collection(imageC)

    # add original geometry to each image
    # we do not need to do this really:
    # imageC = imageC.map(function(im){return(im.set(newDict))});
    # add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({"original_polygon": geom, "WSDA": a_feature})

    # imageC = imageC.sort('system:time_start', True)
    return imageC


def addEVI_to_image_Sentiel(image):
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))",
        {"NIR": image.select("B8"), "RED": image.select("B4"), "BLUE": image.select("B2")},
    ).rename("EVI")
    return image.addBands(evi)


def add_EVI_collection_Sentiel(image_IC):
    EVI_IC = image_IC.map(addEVI_to_image_Sentiel)
    return EVI_IC


def addNDVI_to_image_Sentinel(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def add_NDVI_collection_Sentinel(image_IC):
    NDVI_IC = image_IC.map(addNDVI_to_image_Sentinel)
    return NDVI_IC


def add_system_start_time_image(image):
    return image.addBands(image.metadata("system:time_start").rename("system_start_time"))


def add_system_start_time_collection(colss):
    return colss.map(add_system_start_time_image)


def maskS2clouds(image):
    """Return an image with dropping its cloudy pixels

    Arguments
    ---------
    image : an ee.Image

    Returns
    ---------
    reduced  :  an ee.Image with "no" cloudy pixels
    """
    qa = image.select("QA60")
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

    """
    I have to add "system:time_start" back to the image here.
    The Python version of the code behaves differently from its JS counterpart.
    """

    helper = image.updateMask(mask).divide(10000)
    helper = ee.Image(helper.copyProperties(image, properties=["system:time_start"]))
    # helper.select('SR_B1').multiply(0.0000275).add(-0.2);
    return helper


def feature2ee(file):
    """
        Source: bit.ly/35eGjjO
        full web address:
    bikeshbade.com.np/tutorials/Detail/?title=Geo-pandas%20data%20frame%20to%20GEE%20feature%20collection%20using%20Python&code=13
        Supposed to convert GeoPanda dataframe to ee.featureCollection.
        We use this on Colab for App.
    """

    # Exception handler
    try:
        # check if the file is shapefile or CSV
        if file.endswith(".shp"):
            gdf = gpd.read_file(file, encoding="utf-8")
            g = [i for i in gdf.geometry]
            features = []

            # for Polygon geo data type
            if gdf.geom_type[0] == "Polygon":
                for i in range(len(g)):
                    g = [i for i in gdf.geometry]
                    x, y = g[i].exterior.coords.xy
                    cords = np.dstack((x, y)).tolist()

                    g = ee.Geometry.Polygon(cords)
                    feature = ee.Feature(g)
                    features.append(feature)
                print("done")

                ee_object = ee.FeatureCollection(features)

                return ee_object

            # for LineString geo data type
            elif gdf.geom_type[0] == "LineString":
                for i in range(len(g)):
                    g = [i for i in gdf.geometry]
                    x, y = g[i].exterior.coords.xy
                    cords = np.dstack((x, y)).tolist()
                    double_list = reduce(lambda x, y: x + y, cords)

                    g = ee.Geometry.LineString(double_list)
                    feature = ee.Feature(g)
                    features.append(feature)
                print("done")

                ee_object = ee.FeatureCollection(features)

                return ee_object

            # for Point geo data type
            elif gdf.geom_type[0] == "Point":
                for i in range(len(g)):
                    g = [i for i in gdf.geometry]
                    x, y = g[i].exterior.coords.xy
                    cords = np.dstack((x, y)).tolist()
                    double_list = reduce(lambda x, y: x + y, cords)
                    single_list = reduce(lambda x, y: x + y, double_list)

                    g = ee.Geometry.Point(single_list)
                    feature = ee.Feature(g)
                    features.append(feature)
                print("done")

                ee_object = ee.FeatureCollection(features)

                return ee_object

        # check if the file is shapefile or CSV
        # for CSV we need to have file with X and Y
        elif file.endswith(".csv"):
            df = pd.read_csv(file)
            features = []
            for i in range(df.shape[0]):
                x, y = df.x[i], df.y[i]
                latlong = [x, y]
                g = ee.Geometry.Point(latlong)
                feature = ee.Feature(g)
                features.append(feature)
            print("done")

            ee_object = ee.FeatureCollection(features)
            return ee_object
    except:
        print("Wrong file format")


# def mosaicByDate(imcol):
#     imlist = imcol.toList(imcol.size())

#     def return_image_date(im):
#         return ee.Image(im).date().format("YYYY-MM-dd")

#     unique_dates = imlist.map(return_image_date).distinct()

#     def function2(d):
#         d = ee.Date(d)
#         im = imcol.filterDate(d, d.advance(1, "day")).mosaic()
#         return im.set(

#     "system:time_start", d.millis(),
#     "system:id", d.format("YYYY-MM-dd"))

#     mosaic_imlist = unique_dates.map(function2)

#     return ee.ImageCollection(mosaic_imlist)

# def mosaicByDate(imcol):
#     # imcol: An image collection
#     # returns: An image collection

#     imlist = imcol.toList(imcol.size())
#     def F1(im):
#         return ee.Image(im).date().format("YYYY-MM-dd")

#     unique_dates = imlist.map(F1).distinct()

#     def F2(d):
#         d = ee.Date(d)
#         im = imcol.filterDate(d, d.advance(1, "day")).mosaic()
#         im = im.set("system:time_start", d.millis(), "system:id", d.format("YYYY-MM-dd"))
#         return im

#     mosaic_imlist = unique_dates.map(F2)
#     return ee.ImageCollection(mosaic_imlist)


## Method for mosaicing by date, makes image id the date of image acquisition


def mosaicByDate(imcol):
    """
    source: https://gist.github.com/giswqs/e85b0371bc3cd15ff6ccd5adfa9643d7
    imcol: An image collection
    returns: An image collection
    """
    imlist = imcol.toList(imcol.size())

    def imdate(im):
        return ee.Image(im).date().format("YYYY-MM-dd")

    unique_dates = imlist.map(imdate).distinct()

    def dater(d):
        d = ee.Date(d)
        im = imcol.filterDate(d, d.advance(1, "day")).mosaic()
        return im.set("system:time_start", d.millis(), "system:id", d.format("YYYY-MM-dd"))

    mosaic_imlist = unique_dates.map(dater)
    return ee.ImageCollection(mosaic_imlist)


def copyProps(index):
    # https://gist.github.com/giswqs/e85b0371bc3cd15ff6ccd5adfa9643d7
    source = ee.Image(metadata_col.toList(metadata_col.size()).get(index))
    dest = ee.Image(LS_collection.toList(LS_collection.size()).get(index))
    image = ee.Image(dest.copyProperties(source, properties=["system:time_start"]))
    return image


def split_feacturecollection_to_blocks(fc, simplify=0, block_size=1000, sel_polygon=True):
    blocks = dict()

    def sp_simplify(feature):
        return feature.setGeometry(feature.geometry().simplify(maxError=simplify))

    if simplify > 0:
        fc = fc.map(sp_simplify)

    feature_count = fc.size().getInfo()  # Get the total number of features
    num_blocks = feature_count // block_size  # Calculate the number of blocks
    if feature_count % block_size != 0:
        num_blocks += 1
    if num_blocks > 1:
        print(f"{num_blocks=}")
    # Iterate over the blocks and export to Google Drive
    for block_index in range(num_blocks):
        start_index = block_index * block_size
        end_index = min((block_index + 1) * block_size, feature_count)
        # Get the block of features
        block = fc.toList(block_size, start_index)
        print(f"block:{block_index} block_size:{len(block.getInfo())}")
        tmp_geojson = ee.FeatureCollection(block).getInfo()
        if sel_polygon:
            tmp_geojson = get_polygons_from_geojson(tmp_geojson)
        blocks[block_index] = tmp_geojson
    return blocks


def split_SF_to_blocks(fc, block_size=1000):
    """
    This is a modification of split_large_feacturecollection_to_blocks() function.
    The update is that this function will split a shapefile of type geopandas.geodataframe.GeoDataFrame.
    """

    """Returns a dataframe where data points are interval_size-day apart.
       This function regularizes the data between the minimum and maximum dates
       present in the data. 

    Arguments
    ---------
    SF : Shapefile 
           of type geopandas.geodataframe.GeoDataFrame

    Returns
    -------
    blocks : dictionary
               where each entry is a smaller shapefile
    """

    blocks = dict()

    feature_count = len(fc)  # Get the total number of features
    num_blocks = feature_count // block_size  # Calculate the number of blocks
    if feature_count % block_size != 0:
        num_blocks += 1

    # Iterate over the blocks and export to Google Drive
    for block_index in range(num_blocks):
        start_index = block_index * block_size
        end_index = min((block_index + 1) * block_size, feature_count)
        blocks[block_index] = fc[start_index:end_index]  # Get FC blocks
    return blocks
