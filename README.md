# Citibike, demand prediction

#Start with DataProcessing_Spark.ipynb.

This code was ment to be executed within a "IBM Watson Studio" notebook.
go to https://www.ibm.com/cloud/watson-studio and sign-up for a free trial.

Create a project and set up a spark instance and an cloud object storage instance.
See this video for a tutorial on how to create a project: https://www.youtube.com/watch?v=8-MlqMxFNuc.

Create a notebook and select the runtime environment as the spark instance you created. The defaul name is "spark-da".
Set the Language to Python 3.5. And the Spark version to 2.1.

See this video for a tuturial on how to create a notebook in Watson Studio, make sure you select the options above and not the ones from the video. https://www.youtube.com/watch?v=iNw4tQAJoO4

The raw source files can be downloaded from https://s3.amazonaws.com/tripdata/index.html, all files with patern YYYYMM_citibike-tripdata.csv or YYYYMM_citibikenyc-tripdata.csv .

The files from 2016 and 2017 are training and test data, and data from 2018 is to be used as prediction data.

For this program the files have been merged, leaving only one header, before being compressed using BZ2, and then uploaded to IBM Watson Cloud Storage within the same bucket of your project.

Put all the files in the same folder, where there only them as csv files. Go to that folder and run the commands below:

$head -1 201601-citibike-tripdata.csv > 2016-2017_citibike-tripdata; tail -n +2 -q 2016*-tripdata.csv >> 2016-2017_citibike-tripdata; tail -n +2 -q 2017*-tripdata.csv >> 2016-2017_citibike-tripdata
$bzip2 -zf 2016-2017_citibike-tripdata 2016-2017_citibike-tripdata.csv.bz2

$head -1 201801-citibike-tripdata.csv > 2018Q1_citibike-tripdata; tail -n +2 -q 2018*-tripdata.csv >> 2018Q1_citibike-tripdata
$bzip2 -zf 2018Q1_citibike-tripdata 2018Q1_citibike-tripdata.csv.bz2

The resulting compressed files can be found in this github project under /data.

Upload the file to your Watson project as a Data Asset.

Upload the files "calendar.csv" and "WeatherData_1261418.csv". The calendar file is a simple formula in excel. For the weather file references please refer to the full report and also to http://w2.weather.gov/climate/index.php?wfo=okx .

When you open the notebook, go to the "Data" panel and select the file "2016-2017_citibike-tripdata.csv.bz2" and click the "insert to code" option and next select "Insert Credentials". It will create a dictionary like the one below:

# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials = {
    'IBM_API_KEY_ID': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'IAM_SERVICE_ID': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
    'BUCKET': 'pilotbigdataprojectxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'FILE': '2016-2017_citibike-tripdata.csv.bz2'
}

Get the values from this dictionary and manually overwrite the one that already exist, keep the existing variable and key names.

Optionally change the stationId and testRunInd variables to reduce the dataset.
