import os
import pandas as pd
import json
import uuid

import scipy as sp
import time

def create_unique_id():
    return time.time() + sp.rand()

def parsePerfLogCallstack(rootEntry, gid, time, CMID, UID, URL, RQT, PQ, CPU, SQLT):
    records = []
    level = 1
    #print("start to parse perflog call stack ", rootEntry)
    if rootEntry["n"] and rootEntry["i"] and rootEntry["t"]:
        record = {}
        name = rootEntry["n"].replace('"','')
        record["name"] = name
        record["i"] = rootEntry["i"]
        record["t"] = rootEntry["t"]
        record["GID"] = gid
        record["time"] = time
        record["CMID"] = CMID
        record["UID"] = UID
        record["URL"] = URL
        record["RQT"] = RQT
        record["PQ"] = PQ
        record["CPU"] = CPU
        record["SQLT"] = SQLT

        record["totalTime"] = rootEntry["t"]
        record["parent"] = ""
        rootId = create_unique_id()
        record["uid"] = rootId
        records.append(record)

        unknownTime = rootEntry["t"]
        if rootEntry["sub"] :
            nextLevel = level + 1
            for subEntry in rootEntry["sub"]:
                parseEntry(records, subEntry, rootEntry["t"], nextLevel, gid, name)
                unknownTime = unknownTime - subEntry["t"]

            if unknownTime < rootEntry["t"] and unknownTime > 0:
                otherrecord = {}
                otherrecord["name"] = "others"
                otherrecord["i"] = ''
                otherrecord["GID"] = gid
                otherrecord["t"] = unknownTime
                otherrecordId = create_unique_id()
                otherrecord["uid"] = otherrecordId
                otherrecord["parent"] = name
                otherrecord["totalTime"] = rootEntry["t"]
                records.append(otherrecord)

    #print("records ", records)
    return records

def parseEntry(records, entry, totalTime, level, gid, parentName):
    #print("start to parseEntry")
    if entry["n"] and entry["i"] and entry["t"]:
        record = {}
        name = entry["n"].replace('"','')
        record["name"] = name
        record["i"] = entry["i"]
        record["t"] = entry["t"]
        record["GID"] = gid
        record["totalTime"] = totalTime
        record["level"] = level
        record["parent"] = parentName
        recordId = create_unique_id()
        record["uid"] = recordId
        records.append(record)

    unknownTime = entry["t"]

    if "sub" in entry.keys():
        nextLevel = level+1

        for subEntry in entry["sub"]:
            parseEntry(records, subEntry, totalTime, nextLevel, gid, name)
            unknownTime = unknownTime - subEntry["t"]

        if unknownTime < entry["t"] and unknownTime > 0:
            record = {}
            record["name"] = "others"
            record["i"] = ""
            record["t"] = unknownTime
            record["GID"] = gid
            record["totalTime"] = totalTime
            record["level"] = nextLevel
            record["parent"] = name

            recordId = create_unique_id()
            record["uid"] = recordId

            records.append(record)
    










