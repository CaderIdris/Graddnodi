from pathlib import Path

from jinja2 import Environment, FileSystemLoader

toml_path = Path("senseurcity.toml")

operator = "mean"
period = "1m"

reference_fields = {
    "NO": {
        "Field Name": "NO"
    },
    "NO2": {
        "Field Name": "NO2"
    },
#    "O3": {
#        "Field Name": "O3"
#    },
    "CO": {
        "Field Name": "CO"
    },
    "CO2": {
        "Field Name": "CO2"
    },
    "PM1": {
        "Field Name": "PM1"
    },
    "PM2.5": {
        "Field Name": "PM2.5"
    },
    "PM10": {
        "Field Name": "PM10"
    }
}

device_fields = {
    "P": {
        "Field Name": "BMP280",
        "Boolean Filters": {
            "BMP280_flag": "None",
        }
    },
    "T": {
        "Field Name": "SHT31TI",
        "Boolean Filters": {
            "SHT31TI_flag": "None",
        }
    },
    "RH": {
        "Field Name": "SHT31HI",
        "Boolean Filters": {
            "SHT31HI_flag": "None",
        }
    },
    "NO": {
        "Field Name": "NO_B4_P1",
        "Boolean Filters": {
            "NO_B4_P1_flag": "None",
        }
    },
    "NO2": {
        "Field Name": "NO2_B43F_P1",
        "Boolean Filters": {
            "NO2_B43F_P1_flag": "None",
        }
    },
#    "O3": {
#        "Field Name": "OX_A431_P1",
#        "Boolean Filters": {
#            "OX_A431_P1_flag": "None",
#        }
#    },
    "CO": {
        "Field Name": "CO_A4_P1",
        "Boolean Filters": {
            "CO_A4_P1_flag": "None",
        }
    },
    "CO2": {
        "Field Name": "D300",
        "Boolean Filters": {
            "D300_flag": "None",
        }
    },
    "PM1": {
        "Field Name": "OPCN3PM1",
        "Boolean Filters": {
            "OPCN3PM1_flag": "None"
        }
    },
    "PM2.5": {
        "Field Name": "OPCN3PM25",
        "Boolean Filters": {
            "OPCN3PM25_flag": "None"
        }
    },
    "PM10": {
        "Field Name": "OPCN3PM10",
        "Boolean Filters": {
            "OPCN3PM10_flag": "None"
        }
    },
}

studies = {
    "ANT_REF_R801": {
        "Study": {
            "Name": "ANT_REF_R801",
            "Start": "2020-01-21",
            "End": "2021-07-21",
            "Operator": operator,
            "Period": period,
            "Reference": {
                "Bucket": "SensEURCity",
                "Measurement": "Reference"
            },
            "Devices": {
                "Antwerp_4049A6": {
                    "Name": "Antwerp_4049A6",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4047DD": {
                    "Name": "Antwerp_4047DD",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_40623F": {
                    "Name": "Antwerp_40623F",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4047E7": {
                    "Name": "Antwerp_4047E7",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_402B00": {
                    "Name": "Antwerp_402B00",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4047E0": {
                    "Name": "Antwerp_4047E0",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_406249": {
                    "Name": "Antwerp_406249",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_40499C": {
                    "Name": "Antwerp_40499C",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-10-01"
                },
                "Antwerp_40642B": {
                    "Name": "Antwerp_40642B",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4065E3": {
                    "Name": "Antwerp_4065E3",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_408175": {
                    "Name": "Antwerp_408175",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_40499F": {
                    "Name": "Antwerp_40499F",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4067B3": {
                    "Name": "Antwerp_4067B3",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4043AE": {
                    "Name": "Antwerp_4043AE",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4065EA": {
                    "Name": "Antwerp_4065EA",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_408165": {
                    "Name": "Antwerp_408165",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4065DA": {
                    "Name": "Antwerp_4065DA",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4067B0": {
                    "Name": "Antwerp_4067B0",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_406246": {
                    "Name": "Antwerp_406246",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4065DD": {
                    "Name": "Antwerp_4065DD",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_40641B": {
                    "Name": "Antwerp_40641B",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4067BA": {
                    "Name": "Antwerp_4067BA",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4065E0": {
                    "Name": "Antwerp_4065E0",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_406424": {
                    "Name": "Antwerp_406424",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_402723": {
                    "Name": "Antwerp_402723",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4067BD": {
                    "Name": "Antwerp_4067BD",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4047CD": {
                    "Name": "Antwerp_4047CD",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4065D0": {
                    "Name": "Antwerp_4065D0",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4047D7": {
                    "Name": "Antwerp_4047D7",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4043B1": {
                    "Name": "Antwerp_4043B1",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-10-01"
                },
                "Antwerp_408178": {
                    "Name": "Antwerp_408178",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_408168": {
                    "Name": "Antwerp_408168",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4043A7": {
                    "Name": "Antwerp_4043A7",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
                "Antwerp_4065D3": {
                    "Name": "Antwerp_4065D3",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "2020-07-01"
                },
            }
        },
        "Reference Fields": reference_fields,
        "Device Fields": device_fields
    },
    "OSL_REF_KVN": {
        "Study": {
            "Name": "OSL_REF_KVN",
            "Start": "2020-01-21",
            "End": "2021-07-21",
            "Operator": operator,
            "Period": period,
            "Reference": {
                "Bucket": "SensEURCity",
                "Measurement": "Reference"
            },
            "Devices": {
                "Oslo_651EF5": {
                    "Name": "Oslo_651EF5",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64FD11": {
                    "Name": "Oslo_64FD11",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_65063E": {
                    "Name": "Oslo_65063E",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_425FB3": {
                    "Name": "Oslo_425FB3",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64B082": {
                    "Name": "Oslo_64B082",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_647D5A": {
                    "Name": "Oslo_647D5A",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64CB6D": {
                    "Name": "Oslo_64CB6D",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_426178": {
                    "Name": "Oslo_426178",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64FD0A": {
                    "Name": "Oslo_64FD0A",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_652FA4": {
                    "Name": "Oslo_652FA4",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_652D3A": {
                    "Name": "Oslo_652D3A",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_65325E": {
                    "Name": "Oslo_65325E",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2021-03-14"
                },
                "Oslo_40642E": {
                    "Name": "Oslo_40642E",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_40458D": {
                    "Name": "Oslo_40458D",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_40817F": {
                    "Name": "Oslo_40817F",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_4065ED": {
                    "Name": "Oslo_4065ED",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "7d"
                },
                "Oslo_65326C": {
                    "Name": "Oslo_65326C",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_652FAF": {
                    "Name": "Oslo_652FAF",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_649312": {
                    "Name": "Oslo_649312",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64E9C5": {
                    "Name": "Oslo_64E9C5",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64CB70": {
                    "Name": "Oslo_64CB70",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_426179": {
                    "Name": "Oslo_426179",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_40816F": {
                    "Name": "Oslo_40816F",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_425FB4": {
                    "Name": "Oslo_425FB4",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_653257": {
                    "Name": "Oslo_653257",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64A291": {
                    "Name": "Oslo_64A291",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_652A32": {
                    "Name": "Oslo_652A32",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_648B91": {
                    "Name": "Oslo_648B91",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64CB78": {
                    "Name": "Oslo_64CB78",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_6517DD": {
                    "Name": "Oslo_6517DD",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_64A292": {
                    "Name": "Oslo_64A292",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_651EFC": {
                    "Name": "Oslo_651EFC",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
                "Oslo_649526": {
                    "Name": "Oslo_649526",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "2020-11-14"
                },
            }
        },
        "Reference Fields": reference_fields,
        "Device Fields": device_fields
    },
    "ZAG_REF_IMI": {
        "Study": {
            "Name": "ZAG_REF_IMI",
            "Start": "2020-01-21",
            "End": "2021-07-21",
            "Operator": operator,
            "Period": period,
            "Reference": {
                "Bucket": "SensEURCity",
                "Measurement": "Reference"
            },
            "Devices": {
                "Zagreb_648169": {
                    "Name": "Zagreb_648169",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_427907": {
                    "Name": "Zagreb_427907",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2021-01-14"
                },
                "Zagreb_40641E": {
                    "Name": "Zagreb_40641E",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_427906": {
                    "Name": "Zagreb_427906",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_406414": {
                    "Name": "Zagreb_406414",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_64E03B": {
                    "Name": "Zagreb_64E03B",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "7d"
                },
                "Zagreb_428164": {
                    "Name": "Zagreb_428164",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_4047D0": {
                    "Name": "Zagreb_4047D0",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2021-01-14"
                },
                "Zagreb_652FA1": {
                    "Name": "Zagreb_652FA1",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_648157": {
                    "Name": "Zagreb_648157",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_649738": {
                    "Name": "Zagreb_649738",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_42816D": {
                    "Name": "Zagreb_42816D",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_64876C": {
                    "Name": "Zagreb_64876C",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_64C52B": {
                    "Name": "Zagreb_64C52B",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_652A38": {
                    "Name": "Zagreb_652A38",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_64C225": {
                    "Name": "Zagreb_64C225",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                },
                "Zagreb_64876B": {
                    "Name": "Zagreb_64876B",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "2020-10-01"
                }
            }
        },
        "Reference Fields": reference_fields,
        "Device Fields": device_fields
    },
    "Ispra": {
        "Study": {
            "Name": "Ispra",
            "Start": "2020-01-21",
            "End": "2021-07-21",
            "Operator": operator,
            "Period": period,
            "Reference": {
                "Bucket": "SensEURCity",
                "Measurement": "Reference"
            },
            "Devices": {
                "Antwerp_4065E3": {
                    "Name": "Antwerp_4065E3",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "7d"
                },
                "Antwerp_40641B": {
                    "Name": "Antwerp_40641B",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "7d"
                },
                "Antwerp_4065E0": {
                    "Name": "Antwerp_4065E0",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "7d"
                },
                "Antwerp_4065D0": {
                    "Name": "Antwerp_4065D0",
                    "Bucket": "SensEURCity",
                    "Measurement": "Antwerp",
                    "Training Cutoff": "7d"
                },
                "Oslo_40642E": {
                    "Name": "Oslo_40642E",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "7d"
                },
                "Oslo_40458D": {
                    "Name": "Oslo_40458D",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "7d"
                },
                "Oslo_4065ED": {
                    "Name": "Oslo_4065ED",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "7d"
                },
                "Oslo_40816F": {
                    "Name": "Oslo_40816F",
                    "Bucket": "SensEURCity",
                    "Measurement": "Oslo",
                    "Training Cutoff": "7d"
                },
                "Zagreb_406414": {
                    "Name": "Zagreb_406414",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "7d"
                },
                "Zagreb_4047D0": {
                    "Name": "Zagreb_4047D0",
                    "Bucket": "SensEURCity",
                    "Measurement": "Zagreb",
                    "Training Cutoff": "7d"
                }
            }
        },
        "Reference Fields": reference_fields,
        "Device Fields": device_fields
    }
}


sec_vars = {
    "CO": ["RH", "T", "P", "CO2"],
    "CO2": ["RH", "T", "P", "CO"],
    "NO2": ["RH", "T", "P", "CO", "NO"], #"O3", "NO"],
#    "O3": ["RH", "T", "P", "NO", "CO", "NO2"],
    "NO": ["RH", "T", "P", "NO2", "CO"],#, "O3"],
    "PM1": ["RH", "T", "P"],
    "PM2.5": ["RH", "T", "P"],
    "PM10": ["RH", "T", "P"],
}

env = Environment(loader=FileSystemLoader("Settings/"))
template = env.get_template("senseurcity.toml.jinja")
with toml_path.open('w') as file:
    file.write(
        template.render(
            studies=studies,
            sec_vars=sec_vars
        )
    )
