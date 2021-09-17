# -*- coding: utf-8 -*-
from py2neo import Node, Graph, Relationship,NodeMatcher

import os
import pandas as pd
import json
import sys

from parsePerflogCallStack import  *


def connectToNeo4j(url, user, password):
    """建立连接"""
    link = Graph(url, auth=(user, password))

    return link

def deleteAll(link):
    link.delete_all()

def createNode(link, label, name):
    node = Node(label, name=name)
    link.create(node)
    return node

def createRelation(link, nodeA, nodeB):
    relationship = Relationship(nodeA, nodeB)
    link.create(relationship)

#Relationship.

def removeQuotes(string):
    if string.startswith('"'):
        string = string[1:]

    if string.endswith('"'):
        string = string[:-1]
    return string

def process_lines(line, lineNum):
    print("process line")
    records = []
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
        STK = removeQuotes(STK.strip())
        STK = STK.replace('\"\"','\"')
        print("STK:", STK)

        stkData = ""
        try:
            stkData = json.loads(STK)
            records = parsePerfLogCallstack(stkData, GID, _time, CMID, UID, URL, RQT, PQ, CPU, SQLT)
            print("records got ", records)
        except json.decoder.JSONDecodeError:
            print("got json parse error ", stkData)
            print("error msg:", sys.exc_info()[0])

    return records
# create node if not exist
#merge (n:Trace {name:"newMethod",newprop:"newprop"}) return n

#create relationships:
#Match (nodeA:Trace{name:"/acme"}), (nodeB:Trace{name:"/sfapi/v1/soap"}) CREATE (nodeA)-[r:calls{invocations:13, duration:123}]->(nodeB)

#DeleteAll:   MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r
def insertOnePerflogRecordsToNeo4j(records, link):
    if len(records) == 0:
        return
    for eachRecord in records:
        print("each eachRecord ", eachRecord)
        nodename = eachRecord['name']
        if "uid" in eachRecord.keys():
            if "URL" in eachRecord.keys():     #Root of the perf.log
                node = Node("Trace", name=eachRecord['name'], uid=eachRecord['uid'], runtime=eachRecord['time'],
                            GID=eachRecord['GID'], CMID=eachRecord['CMID'], UID=eachRecord['UID'],
                            URL=eachRecord['URL'], RQT=eachRecord['RQT'], PQ=eachRecord['PQ'], CPU=eachRecord['CPU'],
                            SQLT=eachRecord['SQLT'], totalTime=eachRecord['totalTime'], parent=eachRecord['parent'])
            else:
                node = Node("Trace", name=eachRecord['name'], uid=eachRecord['uid'],
                            GID=eachRecord['GID'], parent=eachRecord['parent'])
        else:
            node = Node("Trace", name=nodename, GID=eachRecord['GID'])

        cypherQL = "Merge (n:Trace {name:\"%s\"}) return n " % (nodename)
        # print("cypherQL:", cypherQL)
        link.run(cypherQL)

        nodes = NodeMatcher(link)
        nodeParent = nodes.match("Trace", name=eachRecord['parent']).first()
        if nodeParent:    #if able to find the parent. creat a rel between parent-child
            iteration = eachRecord["i"]
            duration = eachRecord["t"]
            if (not iteration):
                iteration = "1"
            cypherQL = "Match (nodeA:Trace{name:\"%s\"}), (nodeB:Trace{name:\"%s\"}) CREATE (nodeA)-[r:calls{invocations:%s, duration:%s}]->(nodeB)"% (eachRecord['parent'],nodename, str(iteration), str(duration))
            link.run(cypherQL)

def readPerflogCsvNew(csvPath, link):
    print("new Reaf Perflog csv")
    iLineNum = 0
    with open(csvPath, 'r') as file:
        for line in file:
            records = process_lines(line, iLineNum)
            insertOnePerflogRecordsToNeo4j(records, link)
            iLineNum = iLineNum + 1

def createRel(link, nodeAuid,nodeBuid, duration, invocations):
    cypherQL = "Match (a:Trace), (b:Trace) where a.uid = %s AND b.uid = %s CREATE (a) - [r:calls { duration: %s, invocations: %s}] ->(b) return r" % (nodeAuid, nodeBuid, str(duration), str(invocations))
    #print("cypherQL:", cypherQL)
    link.run(cypherQL)
    return

def aggregationTest1(link, invocationCount):
    cypherQL = "Match(aNode:Trace) -[rel:calls]->(bNode:Trace) where rel.invocations > %s return  aNode.name, rel.duration, rel.invocations, bNode.name " % (str(invocationCount))
    print("cypherQL:", cypherQL)
    pdResult = link.run(cypherQL).to_data_frame()

    #for eachOne in result:
    #    print("got one row ", eachOne)
    return pdResult

def aggregationTest2(link, invocationCount):
    cypherQL = "Match(aNode:Trace) -[rel:calls]->(bNode:Trace) where rel.invocations > %s return  aNode.name, bNode.name, sum(rel.duration) as sumduration, sum(rel.invocations) order by sumduration desc limit 10 " % (str(invocationCount))
    print("cypherQL:", cypherQL)
    pdResult = link.run(cypherQL).to_data_frame()

    pdResult.to_csv('top10ExpensiveRels.csv')
    return pdResult

#analysis:
#match (a:Trace)-[r:calls]->(b:Trace) return sum(r.duration) order by sum(r.duration) desc limit 10
#match (a:Trace)-[r:calls]->(b:Trace) return a.name,b.name,sum(r.invocations),sum(r.duration) order by sum(r.duration) desc limit 100

def createIndexByCQL(link):
    cypherQL ="CREATE INDEX on :Trace(GID)"

    result = link.run(cypherQL)
    return

def queryByCQL(link):
    return link.run("MATCH (a:Trace) RETURN a.name").data()

#delete all nodes
#MATCH (n) DETACH DELETE n
if __name__ == '__main__':
    print("Start to run main of read perflog")
    url = "http://localhost:7474"
    user = "neo4j"
    password = "password"
    graphName = "mygraph"
    link = connectToNeo4j(url, user, password)
    deleteAll(link)

    #aliceNode = createNode(link, "person", "Alice")
    #bobNode = createNode(link, "person", "Bob")
    #createRelation(link, aliceNode, bobNode)

    #allRecords = readPerflogCsv("./April21_10mins.csv")
    #allRecords = readPerflogCsv("C:/Users/i052090/Downloads/April28_DC2.csv")
    readPerflogCsvNew("./April21_10mins.csv", link)

    #print(" all records ", allRecords)
    #insertAllRecordsToNeo4j(allRecords, link)

    #createIndexByCQL(link)
    queryTest = queryByCQL(link)

    result = aggregationTest2(link,2)
    #print("test result 2 ", result)







