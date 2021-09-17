import os
import json
import sys

def readInChunks(fileObj, chunkSize=1024):
    while True:
        data = fileObj.read(chunkSize)
        if not data:
            break
        while data[-1:] != '\n':
            data+=fileObj.read(1)
        yield data

def removeQuotes(string):
    if string.startswith('"'):
        string = string[1:]

    if string.endswith('"'):
        string = string[:-1]
    return string

def process_lines(line, lineNum):
    print("process line")
    #parse by comma   ,
    if lineNum == 0:
        print("process header")
    else:
        firstComma = line.index(',')
        if firstComma <0:
            return -1
        _time = line[0:firstComma]
        _time = removeQuotes(_time)
        print("_time:", _time)
        secondComma = line.index(',', firstComma+1)
        if secondComma < 0:
            return -1
        host = line[firstComma+1:secondComma]
        host = removeQuotes(host)
        print("host:", host)
        thirdComma = line.index(',', secondComma+1)
        if thirdComma < 0:
            return -1
        CMID = line[secondComma + 1:thirdComma]
        CMID = removeQuotes(CMID)
        print("CMID:", CMID)

        fourthComma = line.index(',', thirdComma+1)
        if fourthComma < 0:
            return -1
        UID = line[thirdComma + 1:fourthComma]
        UID = removeQuotes(UID)
        print("UID:", UID)

        fifthComman = line.index(',', fourthComma+1)
        if fifthComman < 0:
            return -1
        EID = line[fourthComma + 1:fifthComman]
        EID = removeQuotes(EID)
        print("EID:", EID)

        sixthComman = line.index(',', fifthComman+1)
        if sixthComman < 0:
            return -1
        GID = line[fifthComman + 1:sixthComman]
        GID = removeQuotes(GID)
        print("GID:", GID)

        seventhComman = line.index(',', sixthComman+1)
        if seventhComman < 0:
            return -1
        MTD = line[sixthComman + 1:seventhComman]
        MTD = removeQuotes(MTD)
        print("MTD:", MTD)

        eighthComman = line.index(',', seventhComman+1)
        if eighthComman < 0:
            return -1
        URL = line[seventhComman + 1:eighthComman]
        URL = removeQuotes(URL)
        print("URL:", URL)

        ninethComman = line.index(',', eighthComman+1)
        if ninethComman < 0:
            return -1
        RQT = line[eighthComman + 1:ninethComman]
        RQT = removeQuotes(RQT)
        print("RQT:", RQT)

        tenthComman = line.index(',', ninethComman+1)
        if tenthComman < 0:
            return -1
        MID = line[ninethComman + 1:tenthComman]
        MID = removeQuotes(MID)
        print("MID:", MID)

        eleventhComman = line.index(',', tenthComman+1)
        if eleventhComman < 0:
            return -1
        PID = line[tenthComman + 1:eleventhComman]
        PID = removeQuotes(PID)
        print("PID:", PID)

        twelfthComman = line.index(',', eleventhComman+1)
        if twelfthComman < 0:
            return -1
        PQ = line[eleventhComman + 1:twelfthComman]
        PQ = removeQuotes(PQ)
        print("PQ:", PQ)

        thirteenthComman = line.index(',', twelfthComman+1)
        if thirteenthComman < 0:
            return -1
        MEM = line[twelfthComman + 1:thirteenthComman]
        MEM = removeQuotes(MEM)
        print("MEM:", MEM)

        fourteenthComman = line.index(',', thirteenthComman+1)
        if fourteenthComman < 0:
            return -1
        CPU = line[thirteenthComman + 1:fourteenthComman]
        CPU = removeQuotes(CPU)
        print("CPU:", CPU)

        fifteenthComman = line.index(',', fourteenthComman+1)
        if fifteenthComman < 0:
            return -1
        UCPU = line[fourteenthComman + 1:fifteenthComman]
        UCPU = removeQuotes(UCPU)
        print("UCPU:", UCPU)

        sixteenthComman = line.index(',', fifteenthComman+1)
        if sixteenthComman < 0:
            return -1
        SQLC = line[fifteenthComman + 1:sixteenthComman]
        SQLC = removeQuotes(SQLC)
        print("SQLC:", SQLC)

        seventeenthComman = line.index(',', sixteenthComman+1)
        if seventeenthComman < 0:
            return -1
        SQLT = line[sixteenthComman + 1:seventeenthComman]
        SQLT = removeQuotes(SQLC)
        print("SQLT:", SQLT)

        STK = line[seventeenthComman + 1:]
        STK = removeQuotes(STK)
        print("STK:", STK)

    return 0


if __name__ == '__main__':
    print("start of read huge perflog")

    filePath = "C:/Users/i052090/Downloads/April28_DC2.csv"
    #"_time",host,CMID,UID,EID,GID,MTD,URL,RQT,MID,PID,PQ,MEM,CPU,UCPU,SQLC,SQLT,STK
    #with open(filePath, 'r') as file:
    #    dataOutput = readInChunks(file)

    iLineNum = 0
    with open(filePath, 'r') as file:
        for line in file:
            process_lines(line, iLineNum)
            iLineNum = iLineNum + 1