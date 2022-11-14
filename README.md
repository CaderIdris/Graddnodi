<h1 align="center">
	Graddnodi
</h1>

**Contact**: [CaderIdrisGH@outlook.com](mailto:CaderIdrisGH@outlook.com)

---

This program was designed, tested and run on a Linux machine using Python 3.10. Earlier versions of Python and other operating systems have not been tested.

---

## Table of Contents

1. [Standard Operating Procedure](#standard-operating-procedure)
1. [Requirements](#requirements)
1. [Settings](#settings)
1. [Setup](#setup)
1. [Data Dictionary](#data-dictionary)

---

## Standard Operating Procedure

### Rendering the report

```bash
cd Output/{run_name}/Report 
# Where {run_name} is the name of the run you want to generate the report for
lualatex Report.tex && lualatex Report.tex
# lualatex is run twice to ensure the table of contents is correct
```

---

## Requirements

This program imports data from an InfluxDB 2.x database. Several examples of importing different data formats to an InfluxDB database can be found here. The structure of the data should not matter as the Flux query is dynamically generated in [the config file](./Settings/config.json)

This program generates graphs using the LaTeX backend in matplotlib. The report generated at the end is also in LaTeX format and requires LaTeX to render it to a pdf. 

All python requirements can be found in [requirements.txt](./requirements.txt) and will be installed into a virtual environment when running [setup.sh](./setup.sh)

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

## Setup

---

## Data Dictionary

