#! /usr/bin/python -u
from pysismo.dtaproc import Client
from pysismo.doublesta import TsEvnt

stationinfo = "./info/XJSTA.info"
mseeddir = "../TianShan/MINISEEDData"
sacdir = "./EQPROCE/TrimData/SAC"
responsedir = "./info/POLES_ZEROS_file"
model = "prem"
client = Client(stationinfo=stationinfo,
                mseeddir=mseeddir,
                sacdir=sacdir,
                responsedir=responsedir,
                model=model)

tsevent = TsEvnt("./info/catalog.csv")
tsevent.matchtsevnt(client)
"""
# Data Trim part
epicenter = {
    "minimum": 30,
    "maximum": 40
}
for event in events:
    logger.info("trimming for %s", event['origin'])

    by_event = {"start_offset": 0, "duration": 6000}
    by_phase = {
        "start_ref_phase": ['P', 'p'],
        "start_offset": -100,
        "end_ref_phase": ['PcP'],
        "end_offset": 200
    }
    by_speed = {
            "minimum":2,
            "maximum":5
            }

    client.get_waveform(event, by_speed=by_speed)
"""
