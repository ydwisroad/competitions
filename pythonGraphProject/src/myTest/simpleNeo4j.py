# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from py2neo import Node, Graph, Relationship, NodeMatcher

import os
import pandas as pd
import json

def connectToNeo4j():
    link = Graph("http://localhost:7474", username="neo4j", password="xxxx")
    return link

def parseJsonSimple():
    jsonStr = "{\"data\":{\"sub\":[{\"n\":\"patricc\"}, {\"n\":\"jack\"}],\"powww\":99,\"dev\":69},\"success\":true,\"message\":\"success\"}";
    jsonSsss = json.loads(jsonStr)
    child = jsonSsss['data']
    print("child sub ", child["sub"])

    for each in child["sub"]:
        print(each["n"])

if __name__ == '__main__':
    print("This is the start of simple Neo4j Test")
    #link = connectToNeo4j()
    parseJsonSimple()


