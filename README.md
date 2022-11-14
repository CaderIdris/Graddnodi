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

#### Errors

Errors selects which error calculations to use

|Key|Type|Description|Options|Example|
|---|---|---|---|---|


#### Runtime

Runtime includes all information needed at runtime

|Key|Type|Description|Options|Example|
|---|---|---|---|---|

---

## Setup

---

## Data Dictionary

