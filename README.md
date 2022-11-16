<h1 align="center">
	Graddnodi
</h1>

**Contact**: [CaderIdrisGH@outlook.com](mailto:CaderIdrisGH@outlook.com)

---


## Table of Contents

1. [Introdution](##introduction)
1. [Input](##input)
1. [Output](##output)
1. [Requirements](##requirements)
1. [Setup](##setup)
1. [Standard Operating Procedure](##standard-operating-procedure)
1. [Settings](##settings)
1. [Data Dictionary](##data-dictionary)

---

## Introduction

Graddnodi (en: Calibration) is a program developed as part of a suite of tools designed to analyse the data generated by low-cost air quality sensors and attempt to assess their quality, remove outliers, detect malfunctions and calibrate them over both short distances (collocation) and long distances (experimental pseudo-collocation). 
Graddnodi assesses the calibrates measurements made during a collocation study and assesses their quality via a range of error calculcations.

Graddnodi can output both the results of the calibrations and summary statistics detailing the best performing calibratino techniques and secondary variables to use in multivariate calibrations.
Therefore this program can not only identify the best calibration technique to use for different types of air quality monitors but also allow you to correct the measurements produced by these devices using the best calibration technique.

This program was tested on a data generated during a long-term collocation study performed at the National Physical Laboratory in Teddington. 
Many different air quality monitors were deployed there, both low and high-cost.
They measured a range of pollutants and environmental conditions (e.g temperature, relative humidity) to varying degrees of quality.
The measurements were collected, standardised and uploaded to an InfluxDB 2.x database (using a suite of programs you can find [here](https://github.com/stars/CaderIdris/lists/influxdb-2-x-upload)) before being aggregated in Graddnodi.


---

## Input

---

## Output

---

## Requirements

This program imports data from an InfluxDB 2.x database. Several examples of importing different data formats to an InfluxDB database can be found here. The structure of the data should not matter as the Flux query is dynamically generated in [the config file](./Settings/config.json)

This program generates graphs using the LaTeX backend in matplotlib. The report generated at the end is also in LaTeX format and requires LaTeX to render it to a pdf. 

All python requirements can be found in [requirements.txt](./requirements.txt) and will be installed into a virtual environment when running [setup.sh](./setup.sh)

This program was designed, tested and run on a Linux machine using Python 3.10. Earlier versions of Python and other operating systems have not been tested.

---

## How to Use 

If you are utilising a POSIX compliant shell, you can use the provided source file. By running ```source graddnodi``` or ```. graddnodi``` from your shell in the root of this git repository, you will have access to the following commands:

**run**:

Runs the program

|Flag|Flag|Description|Default|
|---|---|---|---|
|-c|--config-path|Path to the config file|Settings/config.json|
|-i|--influx-path|Path to the influx config|Settings/influx.json|
|-o|--output-path|Where the output is saved to|Output/|
|-f|--full-output|Include full output in saved report (eCDFs, scatter plots, Bland-Altman plots etc)|False, True if called|

**setup**:

Sets up the virtual environment used by Graddnodi.
The virtual environment is optional and you can use your own solution, provided the python interpreter you are using meets all the requirements listed in [Requirements](##requirements).
Setup will not install Python 3.x, the virtual environment pip package or any latex distribution as these require root access.

**help**:

Prints out each command, what they do and the arguments they take.

---

## Setup

You can use the provided source file to setup the Python virtual environment.
Alternatively, you can set up a Python 3 virtual environment yourself using the provided [requirements.txt](./requirements.txt) file to install the required packages.
The Graddnodi source file expects the virtual environment *.env* to be present in the Graddnodi directory, if you name the virtual environment something else or use your systems interpreter then you will have to modify the source file or run the program directly.

The setup command performs the following steps:
```bash
command -v python3 &> /dev/null || (printf 'Python3 cannot be found in PATH\n' && return)
```

This command checks to see if Python 3 is installed.
If it isn't, the user is informed that python3 isn't present in their PATH and the setup is exited.
If python3 is not installed, the user will have to do it themselves. 
The steps required for this very by operating system, version and distribution.
Please consult [the Python website](https://www.python.org) for more information.

```bash
python3 -c 'import venv' &> /dev/null || (printf 'python3 venv is not installed. To set up a virtual environment, please install the python3 venv package\n' && return)
```

This command checks to see if the venv package is installed in the systems interpreter.
If it isn't, a virtual environment cannot be set up. The user is informed of this and the setup exits.
This package can be installed by using Pythons package manager, pip.
(pip install venv or pip3 install venv).

```bash
python3 -m venv .env
```

This command sets up the Python virtual environment in the Graddnodi directory

```bash
.env/bin/pip install -r requirements.txt
```

This command installs all the packages listed in [requirements.txt](./requirements.txt)


---

## Running without the source file

If you are not using the source file for any number of reasons (Using Windows, the fish shell etc) then the following commands can be run to use the program.
Please note that the following commands assume you are using a virtual environment called .env stored in the Graddnodi directory.
If your virtual environment has a different name, has a different path or you are using the system interpreter then you will have to modify the command to point to that instead.

### Running the program

```bash
.env/bin/python3 src/main.py *arguments*
```

|Flag|Flag|Description|Default|
|---|---|---|---|
|-c|--config-path|Path to the config file|Settings/config.json|
|-i|--influx-path|Path to the influx config|Settings/influx.json|
|-o|--output-path|Where the output is saved to|Output/|
|-f|--full-output|Include full output in saved report (eCDFs, scatter plots, Bland-Altman plots etc)|False, True if called|

### Rendering the report

```bash
cd *output folder*/*run name*/Report 
# Where {run name} is the name of the run you want to generate the report for
lualatex Report.tex && lualatex Report.tex
# lualatex is run twice to ensure the table of contents is correct
```

---

## Settings

### config.json

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Devices|dict|See [Devices](####devices)|---|---|
|Calibration|dict|See [Calibration](####calibration)|---|---|
|Errors|dict|See [Errors](####errors)|---|---|
|Runtime|dict|See [Runtime](####runtime)|---|---|
|Debug Stats|boolean|Print debug stats when running program?|true/false|true|

#### Devices

The devices key contains all the devices that will be cross-compared by the program. 
They are added in order of quality, your reference grade instruments should appear at the top, with the presumed lowest quality instruments appearing at the bottom. 
This halves the amount of calculations performed as there's little point in calibrating your highest quality instrument against your lowest quality one.

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|*device name*|dict|See [Device](#####device)|---|---|

##### Device 

The device subheading contains all information needed to query measurements made by a device stored in an InfluxDB 2.x database.

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Bucket|String|The name of the bucket the data is stored in|Valid bucket in InfluxDB 2.x database|"AURN"|
|Measurement|String|The name of the measurement the data is stored under|Valid measurement name within the bucket in the InfluxDB 2.x database|"Automatic Urban Rural Network"|
|Fields|list of dicts|See [Fields](#####fields)|---|---|
|Start|String|Discard all measurements made before this date|Date string in "YYYY/MM/DD HH:MM:SS" format, or "" to include all measurements from the first to End|2019/04/12 15:53:44|
|End|String|Discard all measurements made after this date|Date string in "YYYY/MM/DD HH:MM:SS" format, or "" to include all measurements from Start to the last measurement|2019/05/12 15:53:44|

##### Fields

The fields list contain details of all the different fields monitored by a device that we want to import.

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Tag|String|This tag replaces the field value when data is saved in the program. This standardises the field tag between all devices. Make sure Tag is the same between devices measuring the same field|Any valid string|"PM2.5"|
|Field|String|The field that the data is stored under|Valid field under the measurement in the InfluxDB 2.x database|"Palas Fidas100 pm2.5 ug/m^3"|
|Boolean Filters|dict|See [Boolean Filters](######boolean-filters)|---|---|
|Range Filters|list of dicts|See [Range Filters](######range-filters)|---|---|
|Secondary Filters|list of dicts|See [Secondary Filters](######secondary-filters)|---|---|
|Scaling|list|See [Scaling](######scaling)|---|---|
|Hour Beginning|boolean|When averaging measurements, make timestamp correspond to start of wwindow instead of end|true/false|true|

###### Boolean Filters

The Boolean Filters dict represents all the values to filter the measurements by.
The key represents the key of the tag-set in the database, the value then selects all measurements corresponding to that value.

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|*tag name*|String|Filters out measurements that don't correspond to the tag-set|Any valid tagset under the measurement|"Serial Number": "2750100"|

###### Range Filters

Range Filters is a list containing dicts representing secondary measurements used to filter out primary measurements that lie outside their range.
An example of where this would be useful is if the device you were comparing was mobile but you wanted to isolate the measurements within a specific latitude and longitude.
Boolean filters are applied to all secondary measurements.

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Field|String|The field containing the secondary measurement used to filter the primary measurement|Any valid field under the measurement|"GlobalSat G-Star IV latitude degrees N"|
|Min|float|The minimum value corresponding to a valid primary measurement. If the secondary measurement is lower than this value, the primary measurement is removed|Any float|51.22|
|Max|float|The maximum value corresponding to a valid primary measurement. If the secondary measurement is higher than this value, the primary measurement is removed|Any float|51.44|
|Min Equal|boolean|If the secondary measurement equals the min value, keep the measurement|true/false|true|
|Max Equal|boolean|If the secondary measurement equals the max value, keep the measurement|true/false|true|

###### Secondary Filters

Secondary Fields is a list containing all secondary fields to also be downloaded.
If you want to calibrate using more than one field (e.g temperature, humidity) then add them here.
All boolean filters are applied to secondary measurements.

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Tag|String|Same purpose as Tag under [Devices](#####devices)|The tag you wish to use to represent the secondary measurement|T|
|Field|String|The field containing the secondary measurement|Any valid field under the measurement|Humidity Prescaled [Percent]|
|Scaling|list|See [Scaling](######scaling)|---|---|

###### Scaling

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Start|String|Scale all measurements made after this date|Date string in "YYYY/MM/DD HH:MM:SS" format, or "" to scale all measurements from the first to End|2019/04/12 15:53:44|
|End|String|Scale all measurements made after this date|Date string in "YYYY/MM/DD HH:MM:SS" format, or "" to scale all measurements from Start to the last measurement|2019/05/12 15:53:44|
|Power|int|Power to raise the measurements to|Any valid int|2|
|Slope|float|The slope used to scale the measurements|Any valid float|2.0|
|Offset|float|The value used to offset the measurements|Any valid float|1.1|

#### Calibration

Calibration contains all configurations used for the calibration step. 

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Data|dict|See [Data](#####data)|---|---|
|Techniques|dict|See [Technique](#####technique)|---|---|
|Bayesian Families|dict|See [Bayesian Families](#####bayesian-families)|---|---|

##### Data

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Split|boolean|Split the data in to a training and testing set?|true/false|true|
|Test Size|float|Proportion of data to use for testing|float between 0 and 1.0, not inclusive|0.4|
|Seed|int|Seed used when randomly splitting data into train/test. Allows repeatability|Any int|

##### Technique

This section determines which calibration techniques should be used when calibrating one device against another.

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Ordinary Least Squares|boolean|Use [OLS](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) to calibrate?|true/false|true|
|Ridge|boolean|Use [Ridge Regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) to calibrate?|true/false|true|
|LASSO|boolean|Use [LASSO](https://scikit-learn.org/stable/modules/linear_model.html#lasso) to calibrate?|true/false|true|
|Elastic Net|boolean|Use [Elastic Net](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net) to calibrate?|true/false|true|
|LARS|boolean|Use [Least Angle Regression](https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression) to calibrate?|true/false|true|
|LASSO LARS|boolean|Use [Lasso path using LARS](https://scikit-learn.org/stable/modules/linear_model.html#lars-lasso) to calibrate?|true/false|true|
|Orthogonal Matching Pursuit|boolean|Use [Orthogonal Matching Pursuit](https://scikit-learn.org/stable/modules/linear_model.html#orthogonal-matching-pursuit-omp) to calibrate?|true/false|true|
|RANSAC|boolean|Use [Random Sample Consensus](https://scikit-learn.org/stable/modules/linear_model.html#ransac-regression) to calibrate?|true/false|true|
|Theil Sen|boolean|Use [Theil Sen](https://scikit-learn.org/stable/modules/linear_model.html#theil-sen-regression) to calibrate?|true/false|true|
|Bayesian|boolean|Use [Bayesian](https://www.pymc.io/projects/docs/en/stable/api.html) to calibrate?|true/false|true|

##### Bayesian Families

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Gaussian|boolean|Use a gaussian distribution when doing a bayesian calibration via PyMC|true/false|true|
|Student T|boolean|Use a student t distribution when doing a bayesian calibration via PyMC|true/false|true|
|Bernoulli|boolean|Use a Bernoulli distribution when doing a bayesian calibration via PyMC|true/false|false|
|Beta|boolean|Use a Beta distribution when doing a bayesian calibration via PyMC|true/false|false|
|Binomial|boolean|Use a Binomial distribution when doing a bayesian calibration via PyMC|true/false|false|
|Gamma|boolean|Use a Gamma distribution when doing a bayesian calibration via PyMC|true/false|false|
|Negative Binomial|boolean|Use a Negative Binomial distribution when doing a bayesian calibration via PyMC|true/false|false|
|Poisson|boolean|Use a Poisson distribution when doing a bayesian calibration via PyMC|true/false|false|
|Inverse Gaussian|boolean|Use a Inverse Gaussian distribution when doing a bayesian calibration via PyMC|true/false|false|


#### Errors

Errors selects which error calculations to use

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Explained Variance Score|boolean|Calculate the [explained variance score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Max Error|boolean|Calculate the [max error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Mean Absolute Error|boolean|Calculate the [mean absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Root Mean Squared Error|boolean|Calculate the [root mean squared error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Root Mean Squared Log Error|boolean|Calculate the [root mean squared log error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Median Absolute Error|boolean|Calculate the [median absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Mean Absolute Percentage Error|boolean|Calculate the [mean absolute percentage error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|r2|boolean|Calculate the [r2](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Mean Poisson Deviance|boolean|Calculate the [mean poisson deviance](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Mean Gamma Deviance|boolean|Calculate the [mean gamma deviance](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Mean Tweedie Deviance|boolean|Calculate the [mean tweedie deviance](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|
|Mean Pinball Loss|boolean|Calculate the [mean pinbal loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html) for the predicted measurement vs the reference measurement in the test set|true/false|true|


#### Runtime

Runtime includes all information needed at runtime

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|Name|String|Name of the run. Runs are split in the output folder based on name|Any valid folder name|2022 Collocation|
|Start|String|Date to download measurements from. Devices that didn't start measuring until after that date will have gaps filled with NaN|String representing date in YYYY/MM/DD format|2018/10/01|
|End|String|Date to stop downloading measurements at|String representing date in YYYY/MM/DD format|2020/10/01|
|Average Operator|String|Average operator that InfluxDB will use to average measurement windows|Any valid aggregator function listed [here](https://docs.influxdata.com/flux/v0.x/tags/aggregates/)|mean|
|Averaging Period|String|Period to average data over|Any valid aggregation time period used by InfluxDB 2.x (More info [here](https://docs.influxdata.com/flux/v0.x/stdlib/universe/stateduration/#unit))|1h|

#### Example config.json file

```json
{
	"Devices": {
		"Bushy Park OPC": {
			"Bucket": "AURN",
			"Measurement": "Automatic Urban Rural Network",
			"Fields": [
				{
					"Tag": "PM2.5",
					"Field": "PM2.5 particulate matter",
					"Boolean Filters": {
						"Site Name": "London Teddington Bushy Park",
						"PM2.5 particulate matter unit": "ugm-3 (Ref.eq)"
					},
					"Range Filters": [],
					"Secondary Fields": [],
					"Hour Beginning": true,
					"Scaling": []
				}
			],
			"Start": "",
			"Stop": ""
		},
		"AirView Reference A": {
			"Bucket": "AirView",
			"Measurement": "AirView",
			"Fields": [
				{
					"Tag": "PM2.5",
					"Field": "Palas Fidas100 pm2.5 ug/m^3",
					"Boolean Filters": {
						"Car": "27522",
						"Palas Fidas100 Status": "5x5"
					},
					"Range Filters": [
						{
							"Field": "GlobalSat G-Star IV latitude degrees N",
							"Min": 51.22,
							"Max": 51.44,
							"Min Equal": false,
							"Max Equal": false
						},
						{
							"Field": "GlobalSat G-Star IV longitude degrees E",
							"Min": -0.36,
							"Max": -0.34,
							"Min Equal": false,
							"Max Equal": true
						}
					],
					"Secondary Fields": [],
					"Hour Beginning": true,
					"Scaling": [
						{
							"Start": "2019/04/12 15:53:44",
							"End": "",
							"Power": 1,
							"Slope": 1000,
							"Offset": 0
						}
					]
				}
			],
			"Start": "",
			"Stop": ""
		},
		"NPL AQMesh A": {
			"Bucket": "ACOEM Data",
			"Measurement": "ACOEM UK Systems",
			"Fields": [
				{
					"Tag": "PM2.5",
					"Field": "Particulate Matter (PM 2.5) PreScaled [Micrograms Per Cubic Meter]",
					"Boolean Filters": {
						"SerialNumber": "1505150",
						"Particulate Matter (PM 2.5) Flag": "Valid"
					},
					"Range Filters": [],
					"Secondary Fields": [
						{
							"Tag": "RH",
							"Field": "Humidity PreScaled [Percent]",
							"Scaling": []
						},
						{
							"Tag": "T",
							"Field": "Temperature PreScaled [Celsius]",
							"Scaling": []
						},
						{
							"Tag": "RH$^2$",
							"Field": "Humidity PreScaled [Percent]",
							"Scaling": [
								{
									"Start": "",
									"End": "",
									"Power": 2,
									"Slope": 1,
									"Offset": 0
								}
							]
						},
						{
							"Tag": "T$^2$",
							"Field": "Temperature PreScaled [Celsius]",
							"Scaling": [
								{
									"Start": "",
									"End": "",
									"Power": 2,
									"Slope": 1,
									"Offset": 0
								}
							]
						}
					],
					"Hour Beginning": true,
					"Scaling": []
				}
			],
			"Start": "",
			"Stop": ""
		},
	},
	"Calibration": {
		"Data": {
			"Split": true,
			"Test Size": 0.4,
			"Seed": 72
		},
		"Techniques": {
			"Ordinary Least Squares": true,
			"Ridge": true,
			"LASSO": true,
			"Elastic Net": true,
			"LARS": true,
			"LASSO LARS": true,
			"Orthogonal Matching Pursuit": true,
			"RANSAC": true,
			"Theil Sen": true,
			"Bayesian": true
		},
		"Bayesian Families": {
			"Gaussian": true,
			"Student T": true,
			"Bernoulli": false,
			"Beta": false,
			"Binomial": false,
			"Gamma": false,
			"Negative Binomial": false,
			"Poisson": false,
			"Inverse Gaussian": false
		}
	},
	"Errors": {
		"Explained Variance Score": true,
		"Max Error": true,
		"Mean Absolute Error": true,
		"Root Mean Squared Error": true,
		"Root Mean Squared Log Error": false,
		"Median Absolute Error": true,
		"Mean Absolute Percentage Error": true,
		"r2": true,
		"Mean Poisson Deviance": false,
		"Mean Gamma Deviance": false,
		"Mean Tweedie Deviance": false,
		"Mean Pinball Loss": true
	},
	"Runtime": {
		"Name": "Default",
		"Start": "2018/10/01",
		"End": "2020/10/01",
		"Average Operator": "mean",
		"Averaging Period": "1h"
	},
	"Debug Stats": true
}

```

### influx.json

|Key|Type|Description|Options|Example|
|---|---|---|---|---|
|IP|String|IP address where InfluxDB 2.x database is hosted|IP Address (localhost if hosted on same machine program is being run)|194.60.38.230|
|Port|String|Port InfluxDB communicates on|Valid port (default is 8086)|8086|
|Token|String|Authentication token used by user to upload/download data|Valid authentication token|thisisnotarealtoken|
|Organisation|String|Organisation the user belongs to|Valid organisation|The Government|

#### Example influx.json file

```json
{
	"IP": "194.60.38.230",
	"Port": "8086",
	"Token": "thisisnotarealtoken",
	"Organisation": "The Government"
}
```

---

## Data Dictionary

Graddnodi saves all information it needs to SQLite3 databases.

### Measurements 

The measurements downloaded from InfluxDB 2.x are saved in Output/*run name*/*field*.db.
They are separated into different fields.
Each table in the field represents a device.
Each row in the table corresponds to a single measurement made by a device.

|Name|Definition|Data type|Possible values|Required?|
|---|---|---|---|---|
|Datetime|Timestamp for measurement in YYYY-MM-DD HH:MM:SS+Z|Text|2018-10-01 00:00:00+00:00, 2020-09-30 23:00:00+00:00, etc|Yes|
|Values|Primary measurement values recorded by device|Real|4.636, 5.0198, NULL|No|
|*Secondary Measurement*|Secondary measurements recorded by device|Real|63.66, 22.10, NULL|No|

### Coefficients

The coefficients obtained from the different calibrations are saved in Output/*run name*/*field*/*A vs B*.db.
Each table in the field represents a different calibration technique. 
Rows correspond to different combinations of secondary measurements (e.g x, x + RH, x + T$^2$)

|Name|Definition|Data type|Possible values|Required?|
|---|---|---|---|---|
|index|The combination of variables used in the calibration|Text|x, x + RH, x + T$^2$ etc|Yes|
|coeff.x|The coefficients to scale x measurements by|Real|0.667|Yes|
|sd.x|Standard deviation on coeff.x, only calculated when using gaussian methods|Real|0.040, NULL|No|
|i.Intercept|The intercept to offset the calibrated measurement by|Real|2.067|Yes|
|sd.Intercept|Standard deviation on i.Intercept, only calculated when using gaussian methods|Real|0.079, NULL|No|
|coeff.*secondary measurement*|Coefficient used to scale *secondary measurement* e.g. coeff.T|Real|0.095, NULL|No|
|sd.*secondary measurement*|Standard deviation on coeff.*secondary measurement*, only calculated when using gaussian methods e.g. sd.T|Real|0.013, NULL|No|

### Results 

The results of the error calculations performed on the calibrated test set are saved in Output/_run name_/_field_/_comparison_/_Technique_/Results.db

### Summary Stats

