# -*- coding: utf-8 -*-

###############################################################################
# Description:
#    This is a collection of utility / helper functions.
#    Note that most of these functions are not well tested, but are 
#    prototyping implementations. Also, this is my first time working with 
#    python, so things aren't as optimized as they could be.
#
# Typical meaning of variable names:
#    lines,strings = list of strings
#    line,string   = single string
#    xmlString     = string with xml tags
#    table         = 2D row/column matrix implemented using a list of lists
#    row,list1D    = single row in a table, i.e. single 1D-list
#    rowItem       = single item in a row
#    list1D        = list of items, not necessarily strings
#    item          = single item of a list1D
#    slotValue     = e.g. "terminator" in: play <movie> terminator </movie>
#    slotTag       = e.g. "<movie>" or "</movie>" in: play <movie> terminator </movie>
#    slotName      = e.g. "movie" in: play <movie> terminator </movie>
#    slot          = e.g. "<movie> terminator </movie>" in: play <movie> terminator </movie>
#
# TODO:
#       - change when possible to use list comprehensions
#       - second utilities function with things that require external libraries (numpy, scipy, ...)
#       - dedicated class for xmlString representation and parsing.
#       - need to verify that all functions read/write/expect strings in utf-8 format
###############################################################################



#import os, time
import random, os, re, copy, sys, collections, pickle, pdb, stat, codecs, xmltodict #, matplotlib as plt
import matplotlib.pyplot as plt
from itertools import chain
from math import *
#if not on azure: import xmltodict



#################################################
#variable definitions
#################################################
questionWords = {}
questionWords["dede"] = ["wer", "wie", "was", "wann", "wo", "warum", "woher", "wen", "wem", "wohin", "wieso", "welche", "wieviel"]
questionWords["frfr"] = ["est-ce", "est - ce", "est ce", "estce", "quand", "pourquoi", "quel", "quelle", "que", "qui", "ou", "où", "combien", "comment", "quand"]
questionWords["eses"] = ["qué", "que", "quién", "quien", "quiénes", "quienes", "cuándo", "cuando", "cómo", "como", "dónde", "donde", "por qué", "por que", "cuánto", "cuanto", "cuántos", "cuantos", "cuántas", "cuantas", "cuánta", "cuanta" "cuál", "cual", "cuáles", "cuales", "cuál", "cual"]
questionWords["itit"] = ["chi", "che", "chiunque", "dove", "perché", "perche", "qualcuno", "quale", "quando", "quanto"]  #"come"=="how"
questionWords["ptbr"] = ["aonde", "onde", "quando", "quanto", "quantos", "que", "quê", "quem", "porque", "qual", "quais", "como", "cade", "pode"]
questionWords["ptpt"] = questionWords["ptbr"]


#normalization for different languages. preserves the character counts.
textNormalizationLUT_dede = dict([ ["Ä","A"], ["Ö","O"], ["Ü","U"], ["ä","a"], ["ö","o"], ["ü","u"], ["ß","s"] ])
textNormalizationLUT_frfr = dict([ ["À","A"], ["à","a"], ["Â","A"], ["â","a"], ["Æ","A"], ["æ","a"], ["Ç","C"], ["ç","c"], ["È","E"], ["è","e"], ["É","E"], ["é","e"], ["Ê","E"], ["ê","e"], ["Ë","E"], ["ë","e"], ["Î","I"], ["î","i"], ["Ï","I"], ["ï","i"], ["Ô","O"], ["ô","o"], ["Œ","O"], ["œ","o"], ["Ù","U"], ["ù","u"], ["Û","U"], ["û","u"], ["Ü","U"], ["ü","u"], ["Ÿ","Y"], ["ÿ","y"] ])
textNormalizationLUT_eses = dict([ ["Á","A"], ["É","E"], ["Í","I"], ["Ñ","N"], ["Ó","O"], ["Ú","U"], ["Ü","U"], ["á","a"], ["é","e"], ["í","i"], ["ñ","n"], ["ó","o"], ["ú","u"], ["ü","u"], ["¿","?"], ["¡","!"] ])
textNormalizationLUT_itit = dict([ ["À","A"], ["È","E"], ["É","E"], ["Ì","I"],["Í","I"], ["Î","I"], ["Ò","O"], ["Ó","O"], ["Ù","U"], ["à","a"], ["è","e"], ["é","e"], ["ì","i"], ["í","i"], ["î","i"], ["ò","o"], ["ó","o"], ["ù","u"] ])




#################################################
# file access
#################################################
def readFile(inputFile):
    #reading as binary, to avoid problems with end-of-text characters
    #note that readlines() does not remove the line ending characters
    with open(inputFile,'rb') as f:
        lines = f.readlines()
        #lines = [unicode(l.decode('latin-1')) for l in lines]  convert to uni-code
    return [removeLineEndCharacters(s) for s in lines];


def readBinaryFile(inputFile):
    with open(inputFile,'rb') as f:
        bytes = f.read()
    return bytes


def readFirstLineFromFile(inputFile):
    with open(inputFile,'rb') as f:
        line = f.readline()
    return removeLineEndCharacters(line);
       

#if getting memory errors, use 'readTableFileAccessor' instead
def readTable(inputFile, delimiter='\t', columnsToKeep=None):
    lines = readFile(inputFile);
    if columnsToKeep != None:
        header = lines[0].split(delimiter)
        columnsToKeepIndices = listFindItems(header, columnsToKeep)
    else:
        columnsToKeepIndices = None;
    return splitStrings(lines, delimiter, columnsToKeepIndices)


class readFileAccessorBase:
    def __init__(self, filePath, delimiter):
        self.fileHandle = open(filePath,'rb')
        self.delimiter = delimiter
        self.lineIndex = -1
    def __iter__(self):
        return self
    def __exit__(self, dummy1, dummy2, dummy3):
        self.fileHandle.close()
    def __enter__(self):
        pass
    def next(self):
        self.lineIndex += 1
        line = self.fileHandle.readline()
        line = removeLineEndCharacters(line)
        if self.delimiter != None:
            return splitString(line, delimiter='\t', columnsToKeepIndices=None)
        else:
            return line
        #"This looks wrong: self.index==0 never reached?!"
        #if self.index == 0:
        #    raise StopIteration


#iterator-like file accessor. use e.g. within "for line in readTableFileAccessor("input.txt"):" loop
class readTableFileAccessor(readFileAccessorBase):
    def __init__(self, filePath, delimiter = '\t'):
        readFileAccessorBase.__init__(self, filePath, delimiter)


class readFileAccessor(readFileAccessorBase):
    def __init__(self, filePath):
        readFileAccessorBase.__init__(self, filePath, None)


def writeFile(outputFile, lines, header=None, encoding=None):
    if encoding == None:
        with open(outputFile,'w') as f:
            if header != None:
                f.write("%s\n" % header)
            for line in lines:
                f.write("%s\n" % line)
    else:
        with codecs.open(outputFile, 'w', encoding) as f:  #e.g. encoding=utf-8
            if header != None:
                f.write("%s\n" % header)
            for line in lines:
                f.write("%s\n" % line)


def writeTable(outputFile, table, header=None):
    lines = tableToList1D(table) #better name: convertTableToLines
    writeFile(outputFile, lines, header)


def writeBinaryFile(outputFile, data):
    with open(outputFile,'wb') as f:
        bytes = f.write(data)
    return bytes


def loadFromPickle(inputFile):
    with open(inputFile, 'rb') as filePointer:
         data = pickle.load(filePointer)
    return data


def saveToPickle(outputFile, data):
    p = pickle.Pickler(open(outputFile,"wb"))
    p.fast = True
    p.dump(data)


def makeDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


#removes just the files in the dir, not recursively
def makeOrClearDirectory(directory):
    makeDirectory(directory)
    files = os.listdir(directory)
    for file in files:
        filePath = directory +"/"+ file
        os.chmod(filePath, stat.S_IWRITE )
        if not os.path.isdir(filePath):
            os.remove(filePath)


def removeWriteProtectionInDirectory(directory):
    files = os.listdir(directory)
    for file in files:
        filePath = directory +"/"+ file
        if not os.path.isdir(filePath):
            os.chmod(filePath, stat.S_IWRITE )


def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def deleteAllFilesInDirectory(directory, fileEndswithString):
    for filename in getFilesInDirectory(directory):
        if filename.lower().endswith(fileEndswithString):
            deleteFile(directory + filename)

def getFilesInDirectory(directory, postfix = ""):
    fileNames = [s for s in os.listdir(directory) if not os.path.isdir(directory+"/"+s)]
    if postfix == "":
        return fileNames
    else:
        return [s for s in fileNames if s.lower().endswith(postfix)]

def getDirectoriesInDirectory(directory):
    return [s for s in os.listdir(directory) if os.path.isdir(directory+"/"+s)]



#################################################
# 1D list
#################################################
def isempty(listND):
    if len(listND) == 0:
        return True
    return False


def find(list1D, func):
    return [index for (index,item) in enumerate(list1D) if func(item)]

def listFindItems(list1D, itemsToFind):
    indices = [];
    list1DSet = set(list1D)
    for item in itemsToFind:
        if item in list1DSet:
            index = list1D.index(item) #returns first of possibly multiple hits
            indices.append(index)
    return indices


def listFindItem(list1D, itemToFind):
    index = [];
    if itemToFind in list1D:
        index = list1D.index(itemToFind) #returns first of possibly multiple hits
    return index

#ex: list1D = ['this', 'is', 'a', 'test']; itemToFindList = ['is','a']
def listFindSublist(list1D, itemToFindList1D):
    matchingStartItemIndices = []
    nrItemsInItemToFindList = len(itemToFindList1D)

    for startIndex in range(len(list1D)-nrItemsInItemToFindList+1):
        endIndex = startIndex + nrItemsInItemToFindList -1
        #print list1D[startIndex:endIndex+1]
        if list1D[startIndex:endIndex+1] == itemToFindList1D:
            matchingStartItemIndices.append(startIndex)
    return matchingStartItemIndices


def listExists(stringToFind, strings, ignoreCase=False):
    for string in strings:
        if stringEquals(stringToFind, string, ignoreCase):
            return True
    return False


def listFindSubstringMatches(lines, stringsToFind, containsHeader, ignoreCase):
    indices = []   
    for (index,line) in enumerate(lines):  
        if containsHeader and index==0:
            indices.append(0)
        else:
            for stringToFind in stringsToFind:
                if ignoreCase:
                    stringToFind = stringToFind.upper()
                    line = line.upper()
                if line.find(stringToFind) >= 0:
                    indices.append(index)
                    break
    return indices
 
   
def listSort(list1D, reverseSort=False, comparisonFct=lambda x: x):
    indices = range(len(list1D))
    tmp = sorted(zip(list1D,indices), key=comparisonFct, reverse=reverseSort)
    list1DSorted, sortOrder = map(list, zip(*tmp))
    return (list1DSorted, sortOrder) 


def listExtract(list1D, indicesToKeep):
    indicesToKeepSet = set(indicesToKeep)
    return [item for index,item in enumerate(list1D) if index in indicesToKeepSet]


def listRemove(list1D, indicesToRemove):
    indicesToRemoveSet = set(indicesToRemove)
    return [item for index,item in enumerate(list1D) if index not in indicesToRemoveSet]

def listReverse(list1D):
    return list1D[::-1]

def listRemoveDuplicates(strings):
    newList = []
    newListSet = set()
    newListIndices = []
    for index,string in enumerate(strings):
        if string not in newListSet:
            newList.append(string)
            newListSet.add(string)
            newListIndices.append(index)
    return (newList, newListIndices)


def listRemoveEmptyStrings(strings):
    indices = find(strings, lambda x: x!="")
    return getRows(strings, indices);


def listRemoveEmptyStringsFromEnd(strings):
    while len(strings)>0 and strings[-1] == "":
        strings = strings[:-1]
    return strings


def listIntersection(strings, referenceStrings):
    #return how many items in "strings" also occur in "referenceStrings"
    intersectingStrings = []
    referenceSet = set(referenceStrings)
    for string in strings:
        if string in referenceSet:
            intersectingStrings.append(string)
    return intersectingStrings


def listsIdenticalExceptForPermutation(listA, listB):
    if len(listA) != len(listB):
        return False
    #note: avoid sorting by making this histogram/dictionary based
    listASorted = sorted(listA)
    listBSorted = sorted(listB)
    for (elemA, elemB) in zip(listASorted,listBSorted):
        if elemA!=elemB:
            return False
    return True


def listAverage(numbers):
    return 1.0 *sum(numbers) / len(numbers)


def listProd(numbers):
    product = 1
    for num in numbers:
        product *= num
    return product


    
    
#################################################
# 2D list (e.g. tables)
#################################################
def getColumn(table, columnIndex):
    column = [];
    for row in table:
        column.append(row[columnIndex])
    return column

        
def getRows(table, rowIndices):    
    newTable = [];
    for rowIndex in rowIndices:
        newTable.append(table[rowIndex])
    return newTable


def getColumns(table, columnIndices):    
    newTable = [];
    for row in table:
        rowWithColumnsRemoved = [row[index] for index in columnIndices]
        newTable.append(rowWithColumnsRemoved)
    return newTable    


#creates a longer table by splitting items of a given row
def splitColumn(table, columnIndex, delimiter):
    newTable = [];
    for row in table:
        items = row[columnIndex].split(delimiter)
        for item in items:
            row = list(row) #make copy
            row[columnIndex]=item
            newTable.append(row)
    return newTable


def sortTable(table, columnIndexToSortOn, reverseSort=False, comparisonFct=lambda x: float(x[0])):
    if len(table) == 0:
        return []
    columnToSortOnData = getColumn(table, columnIndexToSortOn)    
    (dummy, sortOrder) = listSort(columnToSortOnData, reverseSort, comparisonFct)
    sortedTable = [];
    for index in sortOrder:
        sortedTable.append(table[index])
    return sortedTable


def removeColumnsFromFileUsingGawk(headerString, columnNamesToKeep, inputFile, outputFile, delimiter='\t'):
    header = headerString.split(delimiter);
    columnIndicesToKeep = listFindItems(header, columnNamesToKeep)
    removeColumnsFromFileUsingGawkGivenIndices(columnIndicesToKeep, inputFile, outputFile)
        
          
def removeColumnsFromFileUsingGawkGivenIndices(columnIndicesToKeep, inputFile, outputFile):
    #Use this function when file is too large to be loaded into memory
    gawkIndicesString = ""
    for index in columnIndicesToKeep:
        gawkIndicesString = gawkIndicesString + " $" + str(index+1) + ","
    gawkIndicesString = gawkIndicesString[:-1]
    gawkCmdString = "gawk -F'\t' 'BEGIN {OFS="+'"\t"'+"} {print" + gawkIndicesString + "}' " + inputFile + " > " + outputFile
    os.popen(gawkCmdString)    


def flattenTable(table):
    return [x for x in reduce(chain, table)]

def tableToList1D(table, delimiter='\t'):
    return [delimiter.join([str(s) for s in row]) for row in table]

# def convertTableToStrings(table, delimiter='\t'):
#     delimiterSeparatedLines = [];
#     for row in table:
#         #row = [str(w) for w in row]
#         delimiterSeparatedLine = delimiter.join(map(str,row))
#         delimiterSeparatedLines.append(delimiterSeparatedLine)
#     return delimiterSeparatedLines


def getNthListElements(list2D, index):
    return [list1D[index] for list1D in list2D]


#map label names to integers
def parseValueKeyTable(valueKeyTable):
    valueToKeyLUT = dict()
    keyToValueLUT = dict()
    for line in valueKeyTable:
        value = int(line[0])
        key = line[1]
        valueToKeyLUT[value] = key
        keyToValueLUT[key] = value
    return(keyToValueLUT, valueToKeyLUT)




#################################################
# ND list (e.g. tables)
#################################################
def endStripList(listND, itemToRemove=''):
    if listND == []:
        return listND
    currPos = len(listND)-1
    while listND[currPos] == itemToRemove:
        currPos -= 1
        if currPos<0:
            break
    return [item for index,item in enumerate(listND) if index <= currPos]






     
#################################################
# string
#################################################
def insertInString(string, pos, stringToInsert):
    return insertInString(string, pos, pos, stringToInsert)


def insertInString(string, textToKeepUntilPos, textToKeepFromPos, stringToInsert):
    return string[:textToKeepUntilPos] + stringToInsert + string[textToKeepFromPos:]


def removeMultipleSpaces(string):
    return re.sub('[ ]+' , ' ', string)


def removeLineEndCharacters(line):
    if line.endswith('\r\n'):
        return line[:-2]
    elif line.endswith('\n'):
        return line[:-1]
    else:
        return line


def replaceNthWord(string, wordIndex, wordToReplaceWith):
    words = string.split()
    words[wordIndex] = wordToReplaceWith
    return " ".join(words)


def removeWords(string, wordsToRemove, ignoreCase=False):
    newWords = []
    for word in string.split():
        if not listExists(word, wordsToRemove, ignoreCase):
            newWords.append(word)
    return " ".join(newWords)


def removeNthWord(string, wordIndex):
    words = string.split()
    if wordIndex == 0:
        stringNew = words[1:]
    elif wordIndex == len(words)-1:
        stringNew = words[:-1]
    else:
        stringNew = words[:wordIndex] + words[wordIndex+1:]
    #stringNew = " ".join(stringNew)
    #stringNew = re.sub('[ \t]+' , ' ', stringNew) #replace multiple spaces or tabs with a single space
    return " ".join(stringNew)


def splitString(string, delimiter='\t', columnsToKeepIndices=None):
    if string == None:
        return None
    items = string.split(delimiter)
    if columnsToKeepIndices != None:
        items = getColumns([items], columnsToKeepIndices)         
        items = items[0]
    return items;


def splitStrings(strings, delimiter, columnsToKeepIndices=None):
    table = [splitString(string, delimiter, columnsToKeepIndices) for string in strings]
    return table; 


def spliceString(string, textToKeepStartPositions, textToKeepEndPositions):
    stringNew = "";
    for (startPos, endPos) in zip(textToKeepStartPositions,textToKeepEndPositions):
        stringNew = stringNew + string[startPos:endPos+1]
    return stringNew


def findFirstSubstring(string, stringToFind, ignoreCase=False):
    if ignoreCase:
        string = string.upper();
        stringToFind = stringToFind.upper();
    return string.find(stringToFind)


def findMultipleSubstrings(string, stringToFind, ignoreCase=False):
    if ignoreCase:
        string = string.upper();
        stringToFind = stringToFind.upper();
    matchPositions = [];
    pos = string.find(stringToFind) 
    while pos >= 0: 
        matchPositions.append(pos) 
        pos = string.find(stringToFind, pos + 1) 
    return matchPositions 


def findMultipleSubstringsInMultipleStrings(string, stringsToFind, ignoreCase=False):
    matches = []
    for (stringToFindIndex,stringToFind) in enumerate(stringsToFind):
        matchStartPositions = findMultipleSubstrings(string, stringToFind, ignoreCase)

        for matchStartPos in matchStartPositions:
            matchEndPos = matchStartPos + len(stringToFind)
            matches.append([matchStartPos,matchEndPos,stringToFindIndex])
    return matches


def findOccurringStringsIndices(string, stringsToFind):
    matchIndices = []
    for (stringToFindIndex,stringToFind) in enumerate(stringsToFind):
        if string.find(stringToFind) >= 0:
            matchIndices.append(stringToFindIndex)
    return matchIndices

def regexMatch(string, regularExpression, matchGroupIndices):  
    regexMatches = re.match(regularExpression, string)
    if regexMatches != None:
        matchedStrings = [regexMatches.group(i) for i in matchGroupIndices]
    else:
        matchedStrings = [None]*len(matchGroupIndices)  
    if len(matchGroupIndices) == 1:
        matchedStrings = matchedStrings[0]
    return matchedStrings


def containsOnlyRegularAsciiCharacters(string):
    return all(ord(c) < 128 for c in string)


#remove all control characters except for TAB
#see: http://www.asciitable.com/
def removeControlCharacters(string):
    chars = [c for c in string if not (ord(c)>=0 and ord(c)<=8)]
    chars = [c for c in string if not (ord(c)>=10 and ord(c)<=31)]
    return "".join(chars)


def stringEquals(string1, string2, ignoreCase=False):
    if ignoreCase:
        string1 = string1.upper()
        string2 = string2.upper()
    return string1 == string2

def ToIntegers(list1D):
    return [int(float(x)) for x in list1D]

def Round(list1D):
    return [round(x) for x in list1D]

def ToFloats(list1D):
    return [float(x) for x in list1D]

def ToStrings(list1D):
    return [str(x) for x in list1D]

#NOTE: could just call function ToIntegers, input format is irrelevant
#def stringsToIntegers(strings):
#    return [int(s) for s in strings]
#def stringsToFloats(strings):
#    return [float(s) for s in strings]
#def floatsToStrings(floats):
#    return [str(f) for f in floats]





#################################################
# xmlString
#    slotValue     = e.g. "terminator" in: play <movie> terminator </movie>
#    slotTag       = e.g. "<movie>" or "</movie>" in: play <movie> terminator </movie>
#    slotName      = e.g. "movie" in: play <movie> terminator </movie>
#    slot          = e.g. "<movie> terminator </movie>" in: play <movie> terminator </movie>
#
#  Note that the functionality around xmlStrings is a bit brittle since some function were
#  written assuming consistent xml tags (e.g. whitespace before '<' tag open characters)
#################################################
def getSlotOpenTag(slotName):
    return "<"+slotName+">"


def getSlotCloseTag(slotName):
    return "</"+slotName+">"


def getSlotTag(slotName, slotValue):
    return getSlotOpenTag(slotName) + " " + slotValue + " " + getSlotCloseTag(slotName)


def normalizeXmlString(xmlString, mode='simple'):
    if mode == 'simple':
        #make sure there is a space before each '<' and after each '>', etc.
        #then remove multiple white spaces, as well as trailing spaces
        xmlString = xmlString.replace('<', ' <')
        xmlString = xmlString.replace('>', '> ')
        xmlString = xmlString.replace('?', ' ? ')
        xmlString = xmlString.replace('!', ' ! ')
        xmlString = xmlString.replace('.', ' . ')
        xmlString = removeMultipleSpaces(xmlString)
        xmlString = xmlString.strip()
    else:
        raise Exception('Mode unknown: ' + mode)
    return xmlString


def isXmlTag(string):
    if parseXmlTag(string) != None:
        return True
    else:
        return False


def parseXmlTag(string):
    isTag = False
    if len(string)>2:
        (tagName, isOpenTag, isCloseTag) = (None, False, False)
        if string[0:2]=="</" and string[-1]==">":
            (isTag, isCloseTag, tagName) = (True, True, string[2:-1])
        elif string[0]=="<" and string[-1]==">":
            (isTag, isOpenTag, tagName) = (True, True, string[1:-1])
    if isTag == True:
        return (tagName, isOpenTag, isCloseTag)
    else:
        return None


def renameSlotName(xmlString, oldSlotName, newSlotName):
    xmlString = xmlString.replace(getSlotOpenTag(oldSlotName), getSlotOpenTag(newSlotName))
    xmlString = xmlString.replace(getSlotCloseTag(oldSlotName), getSlotCloseTag(newSlotName))
    return xmlString


def replaceSlotValues(xmlString, slotNameToReplace, newSlotValue):
    keepLooping = True;
    newXmlString = xmlString
    while keepLooping:
        keepLooping = False;
        slots = extractSlots(newXmlString)
        for slot in slots:
            slotName = slot[1]
            if slotName == slotNameToReplace:
                (slotValueStartPos, slotValueEndPos) = slot[3:5]
                oldXmlString = newXmlString
                newXmlString = insertInString(newXmlString, slotValueStartPos+1, slotValueEndPos, newSlotValue)
                if oldXmlString != newXmlString:
                    keepLooping = True;
                    break #break since start/end positions in "tags" have changed
    return newXmlString


def replaceSlotXmlStringWithSlotName(xmlString, slotNames, slotNamePrefix="SLOT_"):
    newXmlString = xmlString
    for slotName in slotNames:
        #match everything except for the "</" characters
        newXmlString = re.sub("<" + slotName + ">(?:(?!</).)*</" + slotName + ">", slotNamePrefix + slotName.upper(), newXmlString, re.VERBOSE)
    return newXmlString


def slotsFormattedCorrectly(slots, verbose=False):
    if len(slots) % 2 != 0:
        if verbose:
            print "WARNING: odd number of slot open/close tags found: " + str(slots)
        return(False)

    slotNameExpected = None;
    for slot in slots:
        slotName = slot[0]
        isOpenTag = slot[3]
        if (slotNameExpected==None and not isOpenTag):
            if verbose:
                print "WARNING: open tag expected but instead found closing tag: " + str(slots)
            return(False)
        elif (not isOpenTag and slotNameExpected != slotName):
            if verbose:
                print "WARNING: expected closing and opening tag to have same slot name: ", (slotNameExpected,tag)
            return(False)
        if isOpenTag:
            slotNameExpected = slotName
        else:
            slotNameExpected = None
    return(True)


def xmlStringCanBeParsed(xmlString):
    #Note: The MLGTools and/or Bitetools crashes if a hash tag is in the data
    #if xmlString.find("#") >= 0 or xmlString.find("  ") >= 0 or xmlString != xmlString.strip() or not containsOnlyRegularAsciiCharacters(xmlString):
    try:
        extractSlots(xmlString)
        return True
    except:
        return False


def extractSlotsHelper(xmlString, validateSlots=True):
    slotStartPositions = findMultipleSubstrings(xmlString, '<')
    slotEndPositions = findMultipleSubstrings(xmlString, '>')

    #check if all startPositions < endPositions
    if (len(slotStartPositions) != len(slotEndPositions)):
        #assumes no < or > characters in query itself just in tag
        raise Exception("Unequal number of '<' and '>' characters: " + xmlString)
    for (slotStartPos, slotEndPos) in zip(slotStartPositions, slotEndPositions):
        if slotStartPos>slotEndPos:
            raise Exception("Found a '>' before a '<' character: " + xmlString)
        if slotStartPos==slotEndPos-1:
            raise Exception("Found an empty tag (i.e. '<>'): " + xmlString)

    #loop over all tags and add to list
    slots = []
    for (slotStartPos, slotEndPos) in zip(slotStartPositions, slotEndPositions):
        slotName = xmlString[slotStartPos+1:slotEndPos]
        if slotName[0] == '/':
            slotName = slotName[1:]
            boIsOpenTag = False
        else:
            boIsOpenTag = True
        if slotName.find(' ') >= 0:
            raise Exception("Slot names should not contain any whitespaces: " + xmlString)
        slots.append((slotName, slotStartPos, slotEndPos, boIsOpenTag))
        
    #check if identified slots are all formatted correctly
    if validateSlots and slotsFormattedCorrectly(slots)==False:
        raise Exception("Identified slots for |%s| nor formatted correctly: " + str(slots))
    return slots


def extractSlots(xmlString, validateSlots=True):
    newSlots = [];
    slots = extractSlotsHelper(xmlString, validateSlots)

    for (slotIndex,slot) in enumerate(slots):  #only loop over open-tags
        isOpenTag = slot[3]
        if slotIndex % 2 == 0:
            assert(isOpenTag)
            tagOpenSlotName = slot[0]
            tagOpenOuterPos = slot[1]
            tagOpenInnerPos = slot[2]
        else:
            tagCloseSlotName = slot[0]
            assert(not isOpenTag)
            assert(tagOpenSlotName == tagCloseSlotName)
            tagCloseOuterPos = slot[2]
            tagCloseInnerPos = slot[1]
            slotValue = xmlString[tagOpenInnerPos+1:tagCloseInnerPos].strip()
            newSlots.append((slotValue, tagCloseSlotName, tagOpenOuterPos, tagOpenInnerPos, tagCloseInnerPos, tagCloseOuterPos))
    return newSlots


def extractSlotValues(xmlStrings):
    slotValues = {}
    for xmlString in xmlStrings:
        slots = extractSlots(xmlString)
        for slot in slots:
            slotValue = slot[0]
            slotName = slot[1]
            if slotName not in slotValues:
                slotValues[slotName] = []
            slotValues[slotName].append(slotValue)
    return slotValues


def removeTagsFromXmlString(xmlString, slotNamesToKeepOrRemove, keepOrRemove="keep", boRemoveMultipleSpaces=True, boRemovePrePostFixSpaces=True):
    assert(keepOrRemove=="keep" or keepOrRemove=="remove")
    slots = extractSlots(xmlString)
    assert(slots != None) #todo: need to check first if can be parsed, THEN either parse or run code below.
    #if slots == None:
    #    print 'Warning "removeTagsFromXmlString": could not parse sentence. Hence simply removing </, < or > characters" ' + xmlString
    #    xmlStringNoOpenCloseCharacters = xmlString
    #    xmlStringNoOpenCloseCharacters = xmlStringNoOpenCloseCharacters.replace('</','')
    #    xmlStringNoOpenCloseCharacters = xmlStringNoOpenCloseCharacters.replace('<','')
    #    xmlStringNoOpenCloseCharacters = xmlStringNoOpenCloseCharacters.replace('>','')
    #    return xmlStringNoOpenCloseCharacters
        
    textToKeepStartPos = [0]
    textToKeepEndPos = [] 
    for slot in slots:
        slotName = slot[1]
        (tagOpenOuterPos, tagOpenInnerPos, tagCloseInnerPos, tagCloseOuterPos) = slot[2:6]

        if (keepOrRemove=="remove") and (slotName in slotNamesToKeepOrRemove):
            boRemoveTag=True;
        elif (keepOrRemove=="keep") and (slotName not in slotNamesToKeepOrRemove):
            boRemoveTag=True;
        else:
            boRemoveTag = False;
            
        if boRemoveTag:
            textToKeepEndPos.append(tagOpenOuterPos-1)
            textToKeepStartPos.append(tagOpenInnerPos+1)
            textToKeepEndPos.append(tagCloseInnerPos-1)
            textToKeepStartPos.append(tagCloseOuterPos+1)
    textToKeepEndPos.append(len(xmlString)-1)

    #create new string
    xmlStringNew = spliceString(xmlString, textToKeepStartPos, textToKeepEndPos).strip()
    if boRemoveMultipleSpaces:
        xmlStringNew = removeMultipleSpaces(xmlStringNew)
    if boRemovePrePostFixSpaces:
        xmlStringNew = xmlStringNew.strip()

    #sanity check
    slotsNew = extractSlots(xmlStringNew)
    for slot in slotsNew:
        if keepOrRemove=="keep" and slot[1] not in slotNamesToKeepOrRemove:
            pdb.set_trace()
        if keepOrRemove=="remove" and slot[1] in slotNamesToKeepOrRemove:
            pdb.set_trace()
    return xmlStringNew


def removeTagsFromXmlStrings(xmlStrings, slotNamesToKeepOrRemove, keepOrRemove="keep", boRemoveMultipleSpaces=True, boRemovePrePostFixSpaces=True):
    return [removeTagsFromXmlString(s, slotNamesToKeepOrRemove, keepOrRemove, boRemoveMultipleSpaces, boRemovePrePostFixSpaces) for s in xmlStrings]


def removeAllTagsFromXmlString(xmlString, boRemoveMultipleSpaces=True, boRemovePrePostFixSpaces=True):
    slotNamesToRemove = getSlotNameCounts([xmlString]).keys()
    return removeTagsFromXmlString(xmlString, slotNamesToRemove, "remove", boRemoveMultipleSpaces, boRemovePrePostFixSpaces)


def removeAllTagsFromXmlStrings(xmlStrings, boRemoveMultipleSpaces=True, boRemovePrePostFixSpaces=True):
    strings = []
    for xmlString in xmlStrings:
        strings.append(removeAllTagsFromXmlString(xmlString, boRemoveMultipleSpaces, boRemovePrePostFixSpaces))
    return strings


def getSlotNameCounts(xmlStrings):
    if not isinstance(xmlStrings, (list)):
        xmlStrings = [xmlStrings]
    slotCounter = collections.Counter()
    for xmlString in xmlStrings:
        slots = extractSlots(xmlString)
        for slot in slots:
            slotName = slot[1]
            slotCounter[slotName] += 1
    return slotCounter


def getNrSentencesWithSlots(xmlStrings):
    counterTaggedSentences = 0
    for xmlString in xmlStrings:
        slots = extractSlots(xmlString)
        if len(slots) > 0:
            counterTaggedSentences += 1
    return counterTaggedSentences


def getNrSentencesWithoutSlots(xmlStrings):
    return len(xmlStrings) - getNrSentencesWithSlots(xmlStrings)


def findSimilarSlot(slotToFind, slotList,  ignoreSlotValue=True, ignoreSlotName=True):
    if ignoreSlotValue==False or ignoreSlotName==False:
        print "Not supported yet"
    for index,slot in enumerate(slotList):
        #ignore slotValue and slotName, compare for equality: tagOpenOuterPos, tagOpenInnerPos, tagCloseInnerPos, tagCloseOuterPos
        if slotToFind[2:]==slot[2:]:
            return index 
    return -1


def convertXmlStringToIOB(xmlString, addSentenceBeginEndMarkers=False):
    currentLabel = "O"    
    wordLabelPairs = [];

    #making sure each < tag has a leading white-space, and each > has a trailing whitespace.
    #then calling split which uses whitespace as separator.
    words = xmlString.replace('<',' <').replace('>','> ').strip().split()

    for word in words:
        #assert(isXmlTag(word))
        if '<' in word[1:-1] or '>' in word[1:-1]:
            raise Exception("Xml string contains stray '<' or '>' characters: " + xmlString)

        if isXmlTag(word):
            (tagName, isOpenTag, isCloseTag) = parseXmlTag(word)
        else:
            (tagName, isOpenTag, isCloseTag) = (None,None,None)
        if isOpenTag:
            currentLabel = tagName
            writeBeginMarker = True
        elif isCloseTag:
            currentLabel = "O"
        else:
            if currentLabel == "O":
                labelToWrite = currentLabel
            elif writeBeginMarker:
                labelToWrite = "B-"+currentLabel
                writeBeginMarker = False
            else:
                labelToWrite = "I-"+currentLabel
            wordLabelPairs.append([word, labelToWrite])
            
    if addSentenceBeginEndMarkers:        
        wordLabelPairs.insert(0, ["BOS","O"])
        wordLabelPairs.append(["EOS", "O"])             
    return wordLabelPairs


def convertXmlStringsToIOB(xmlStrings):
    iobFormat = []
    for xmlString in xmlStrings:
        wordLabelPairs = convertXmlStringToIOB(xmlString)
        for wordLabelPair in wordLabelPairs:
            iobFormat.append(" ".join(wordLabelPair))
        iobFormat.append("")
    return iobFormat[:-1]


def extractSentencesFromIOBFormat(iobLines):
    sentences = []
    sentence = ""
    for iobLine in iobLines:
        if iobLine =="":
            if sentence != "":
                sentences.append(sentence.strip())
            sentence = ""
        else:
            word = iobLine.split()[0]
            sentence += " " + word
    return sentences


def parseXmlFile(xmlFile):
    s = readFile(xmlFile)
    s = " ".join(s)
    return xmltodict.parse(s)




#################################################
# randomize
#################################################
def randomizeList(listND, containsHeader=False):
    if containsHeader:
        header = listND[0]
        listND = listND[1:]
    random.shuffle(listND)
    if containsHeader:
        listND.insert(0, header)
    return listND


def getRandomListElement(listND, containsHeader=False):
    if containsHeader:
        index = getRandomNumber(1, len(listND)-1)
    else:
        index = getRandomNumber(0, len(listND)-1)
    return listND[index]


def getRandomNumbers(low, high):
    randomNumbers = range(low,high+1)
    random.shuffle(randomNumbers)
    return randomNumbers


def getRandomNumber(low, high):
    randomNumber = random.randint(low,high) #getRandomNumbers(low, high)
    return randomNumber #s[0]


def subsampleList(listND, maxNrSamples):
    indices = range(len(listND))
    random.shuffle(indices)
    nrItemsToSample = min(len(indices), maxNrSamples)
    return [listND[indices[i]] for i in range(nrItemsToSample)]


def randomSplit(list1D, ratio):
    indices = range(len(list1D))
    random.shuffle(indices)
    nrItems = int(round(ratio * len(list1D)))
    listA = [list1D[i] for i in indices[:nrItems]]
    listB = [list1D[i] for i in indices[nrItems:]]
    return (listA,listB)




#################################################
# QEPConsole
#################################################
def parseQEPConsoleOutput(qepOutputFile):
    qepInfo = None
    qepInfos = []
    fileObject = open(qepOutputFile,'r')

    for line in fileObject:
        #check if start of a new query
        if line.startswith("Query: "):
            regularExpression = "Query: \{(.*)\}$"
            query = regexMatch(line, regularExpression, [1])
            if query != None:
                if qepInfo != None:
                    qepInfos.append(qepInfo)
                qepInfo = {}
                qepInfo["query"] = query
                qepInfo["edges"] = []

        #check if Impressions
        elif line.startswith("Impressions"):
            startPos = line.find('[')
            endPos = line.find(']')
            qepInfo["Impressions"] = int(line[startPos+1:endPos])

        #check if edge
        elif line.startswith("Edge = ["):
            regularExpression = "Edge = \[(.*)\]\{UL:(.*)\}$"
            (urlCount,url) = regexMatch(line, regularExpression, [1,2])
            if urlCount != None:
                qepInfo["edges"].append((urlCount,url))
    qepInfos.append(qepInfo)
    return qepInfos





#################################################
# dictionaries
#################################################
def increaseDictValueByOne(dictionary, key, initialValue=0):
    if key in dictionary.keys():
        dictionary[key] += 1;
    else:
        dictionary[key] = initialValue + 1;

def sortDictionary(dictionary, sortIndex=0, reverseSort=False):
    return sorted(dictionary.items(), key=lambda x: x[sortIndex], reverse=reverseSort)

def getDictionary(keys, values, boConvertValueToInt = True):
    dictionary = {}
    for key,value in zip(keys, values):
        if (boConvertValueToInt):
            value = int(value)
        dictionary[key] = value
    return dictionary

def getStringDictionary(keys, values):
    dictionary = {}
    for key,value in zip(keys, values):
        dictionary[key] = value
    return dictionary

def dictionaryToTable(dictionary):
    return (dictionary.items())



#################################################
# collections.Counter()
#################################################
def countFrequencies(list1D):
    frequencyCounts = collections.Counter()
    for item in list1D:
        frequencyCounts[item] += 1
    return frequencyCounts


def countWords(sentences, ignoreCase=True):
    frequencyCounts = collections.Counter()
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if ignoreCase:
                word = word.lower()
            frequencyCounts[word] += 1
    return frequencyCounts


def convertCounterToList(counter, threshold=None):
    sortedKeyValuePairs = counter.most_common()
    if threshold == None:
        return sortedKeyValuePairs
    else:
        newSortedKeyValuePairs = [];
        for keyValuePair in sortedKeyValuePairs:
            if keyValuePair[1] >= threshold:
                newSortedKeyValuePairs.append(keyValuePair)
            else:
                break
        return newSortedKeyValuePairs





#################################################
# confusion matrix
#################################################
def initConfusionMatrix(rowColumnNames):
    confMatrix = {}
    for s in rowColumnNames: #actual
        confMatrix[s] = {}
        for ss in rowColumnNames: #estimated
            confMatrix[s][ss] = 0
    return confMatrix


def printConfusionMatrix(confMatrix, rowColumnNames):
    n = 6
    columnWidth = max(2*n, max([len(s) for s in rowColumnNames]))
    line = "(Row=actual)".ljust(columnWidth)
    for columnName in rowColumnNames:
        line += " | " + columnName.center(n)
    line += " | " + "SUM".center(n)
    print line

    for actualTagName in rowColumnNames:
        rowSum = 0
        line = actualTagName.rjust(columnWidth)
        for estimatedTagName in rowColumnNames:
            value = confMatrix[actualTagName][estimatedTagName]
            rowSum += value
            line += " | " + str(value).center(max(n,len(estimatedTagName)))
        line += " || " + str(rowSum).center(n)
        print line


def plotConfusionMatrix(confMat, title='Confusion matrix', labelNames = None, colorMap=plt.cm.jet, vmin=None, vmax=None):
    plt.imshow(confMat, interpolation='nearest', cmap=colorMap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    if labelNames:
        tick_marks = np.arange(len(labelNames))
        plt.xticks(tick_marks, labelNames, rotation=45, ha='right')
        plt.yticks(tick_marks, labelNames)
    plt.colorbar()
    plt.tight_layout()


def analyseConfusionMatrix(confMatrix, rowColumnNames=None):
    if rowColumnNames == None:
        rowColumnNames = ["pos","neg"]
    posName = rowColumnNames[0]
    negName = rowColumnNames[1]
    tp = confMatrix[posName][posName]
    fp = confMatrix[negName][posName]
    fn = confMatrix[posName][negName]
    return computePrecisionRecall(tp, fp, fn)


#warning: bit hacky, not well tested yet
def analyseConfusionMatrixND(confMatrix, tagNamesSubset=None):
    tp = fp = tn = fn = 0
    confMatrixNames = confMatrix.keys()

    #only compute precision/recall etc for subset of rows/columns
    if tagNamesSubset != None:
        confMatrix = copy.deepcopy(confMatrix)
        for actualTagName in confMatrixNames:
            for estimatedTagName in confMatrixNames:
                if estimatedTagName != "None" and not estimatedTagName in tagNamesSubset:
                    confMatrix[actualTagName]["None"] += confMatrix[actualTagName][estimatedTagName]
                    confMatrix[actualTagName][estimatedTagName] = 0
        for actualTagName in confMatrixNames:
            if not actualTagName=="None" and not actualTagName in tagNamesSubset:
                confMatrix[actualTagName]["None"] = 0

    #compute true positive (tp), true negative (tn), false positive (fp) and false negative (fn)
    for actualTagName in confMatrixNames:
        for estimatedTagName in confMatrixNames:
            if estimatedTagName   == "None" and actualTagName == "None":
                tn += confMatrix[actualTagName][estimatedTagName]
            elif estimatedTagName == "None" and actualTagName != "None":
                fn += confMatrix[actualTagName][estimatedTagName]
            elif estimatedTagName != "None" and actualTagName == estimatedTagName:
                tp += confMatrix[actualTagName][estimatedTagName]
            elif estimatedTagName != "None" and actualTagName != estimatedTagName:
                fp += confMatrix[actualTagName][estimatedTagName]

    (precision,recall) = computePrecisionRecall(tp, fp, fn)
    return (precision, recall, tp, fp, tn, fn)


def getF1Score(precision,recall):
    return 2*precision*recall/(precision+recall)


def computePrecisionRecall(tp, fp, fn):
    if (tp+fp)>0:
        precision = round(100.0 * tp / (tp+fp), 2)
    else:
        precision = -1
    positives = (tp + fn)
    if positives>0:
        recall = round(100.0 * tp / positives, 2)
    else:
        recall = -1
    return (precision,recall)


def computeSinglePrecisionRecall(threshold, groundTruthLabels, classifierScore, weights=None, queries=None):
    #print "*****************************"
    #init
    classifierScore = [float(n) for n in classifierScore]
    if weights != None:
        weights = [float(n) for n in weights]
    else:
        weights = [1] * len(classifierScore)
    if queries == None:
        queries = ["Queries not provided..."] * len(classifierScore)
    assert(len(groundTruthLabels) == len(classifierScore) == len(weights) == len(queries))

    #count true positive and false positives
    p = 0
    n = 0
    tp  = fp  = tn  = fn  = 0
    tpW  = fpW  = tnW  = fnW  = 0
    zeroWeightCounter = 0
    for (query,groundTruthLabel,classificationScore,weight) in zip(queries,groundTruthLabels,classifierScore,weights):

        #if classificationScore == 1:
        #    print groundTruthLabel, classificationScore, weight, query

        #init
        classificationLabel = int(classificationScore>threshold)
        if weight == 0:
            zeroWeightCounter += 1
            continue

        #count number of positives and negatives in test set
        if groundTruthLabel == 1:
            p+=1
        else:
            n+=1

        #compute tp, fp, tn, and fn
        if groundTruthLabel == classificationLabel:
            if classificationLabel == 1:
                tp += 1
                tpW += weight
            elif classificationLabel == 0:
                tn += 1
                tnW += weight
            else:
                error()
            #print "CORRECT: GTLabel=%i, classificationScore=%f, weight=%i, query=%s" % (groundTruthLabel, classificationScore, weight, query)

        else:
            if classificationLabel == 1:
                fp += 1
                fpW += weight
            elif classificationLabel == 0:
                fn += 1
                fnW += weight
            else:
                error()
        #if classificationLabel==1: # and classificationLabel==0:
        #    print "WRONG:   GTLabel=%i, classificationScore=%f, weight=%i, query=%s" % (groundTruthLabel, classificationScore, weight, query)

    #compute p/r
    assert((tp + fn) == p)
    assert((fp + tn) == n)
    precision,recall = computePrecisionRecall(tpW, fpW, fnW)
    #precision = 100.0 * tpW / (tpW + fpW)
    #recall = 100.0 * tpW / (tpW + fnW)
    acc = 100.0 * (tpW + tnW) / (tpW + tnW + fpW + fnW)
    return (precision, recall, acc, tpW, fpW, tnW, fnW, zeroWeightCounter)


def computePrecisionRecallVectors(thresholds, groundTruthLabels, classifierScore, weights=None, queries=None):
    precisionVec = []
    recallVec = []
    accVec = []
    for threshold in thresholds:
        (precision, recall, acc) = computeSinglePrecisionRecall(threshold, groundTruthLabels, classifierScore, weights, queries)[0:3]
        precisionVec.append(precision)
        recallVec.append(recall)
        accVec.append(acc)
    return (precisionVec, recallVec, accVec)


def plotPrecisionRecallCurve(precision, recall):
    area = auc(recall, precision)
    plt.plot(recall, precision, label='Precision-recall curve')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.ylim([-0.02, 1.02])
    plt.xlim([-0.02, 1.02])
    plt.title('AUC=%0.2f' % area)
    #plt.legend(loc="upper right")
    plt.show()




#################################################
# sentence patterns
#################################################
def containsRegexMetaCharacter(string, regexChars = ["\\", "^", "?", ".", "+", "*", "(", ")", "[", "]", "{", "}", "|"]):
    for regexChar in regexChars:
        if string.find(regexChar)>=0:
            return True
    return False


def getRegularExpressionsFromSentencePatterns(sentencePatterns, tagNames, placeHolderFormatString, placeHolderRegEx):
    return [getRegularExpressionFromSentencePattern(s, tagNames, placeHolderFormatString, placeHolderRegEx) for s in sentencePatterns]


def getRegularExpressionFromSentencePattern(sentencePattern, slotNames, placeHolderFormatString, placeHolderRegEx):
    #Note this assumes at most one place holder per sentence pattern
    #Example for a placeHolderRegEx which matches 1-3 words: "((\w+)( \w+){0,2})"
    sentencePatternTag = None
    for tagName in tagNames:
        placeHolder = placeHolderFormatString.format(tagName.upper());
        if sentencePattern.find(placeHolder)<0:
            continue
        sentencePattern = sentencePattern.replace(placeHolder, placeHolderRegEx)
        sentencePattern = removeMultipleSpaces(sentencePattern) + "$"
        sentencePatternTag = tagName              
        break
    assert(sentencePatternTag != None)
    sentencePattern = re.compile(sentencePattern)
    return(sentencePattern, sentencePatternTag)  





#################################################
# processes 
# (start process using: p = subprocess.Popen(cmdStr))
#################################################
def isProcessRunning(processID):
    status = processID.poll();
    if status is None:
        return True
    else:
        return False
        
        
def countNumberOfProcessesRunning(processIDs):
    return sum([isProcessRunning(p) for p in processIDs])
    




#################################################
# python environment
#################################################
def clearAll():
    #not sure if this is working
    sys.modules[__name__].__dict__.clear()
    #all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__") and var != "clearall"]
    #for var in all:
    #    print var
    #    #del globals()[var]
    #    del var




                
#################################################
# arguments
#################################################
def printParsedArguments(options):
    print "Arguments parsed in using the command line:"
    for varName in [v for v in dir(options) if not callable(getattr(options,v)) and v[0] != '_']:
        exec('print "   %s = "') % varName
        exec('print options.%s') % varName


def optionParserSplitListOfValues(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
    




#################################################
# url
################################################# 
def removeURLPrefix(url, urlPrefixes = ["https://www.", "http://www.", "https://", "http://", "www."]):
    for urlPrefix in urlPrefixes:
        if url.startswith(urlPrefix):
            url = url[len(urlPrefix):]
            break
    return url
            
            
def urlsShareSameRoot(url, urlRoot):
    url = removeURLPrefix(url)
    urlRoot = removeURLPrefix(urlRoot)
    if url.startswith(urlRoot):
        return True
    else:
        return False





#################################################
# numpy
#################################################
#def convertListToNumPyArray(list1D, delimiter, columnIndices=None):
#    table = []
#    for line in list1D:
#        row = line.split(delimiter)
#        if columnIndices != None:
#            row = [row[i] for i in columnIndices]
#        table.append(row)
#    return np.array(table);



       
#################################################
# other
#################################################
def printProgressMsg(msgFormatString, currentValue, maxValue, modValue):
    if currentValue % modValue == 1:
        text = "\r"+msgFormatString.format(currentValue, maxValue) #"\rPercent: [{0}] {1}% {2}".format("#"*block + "-"*(barLength-block), round(progress*100,2), status)
        sys.stdout.write(text)
        sys.stdout.flush()


def displayProgressBarPrompt(progress, status = ""):
    barLength = 30
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#"*block + "-"*(barLength-block), round(progress*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def numToString(num, length, paddingChar = '0'):
    if len(str(num)) >= length:
        return str(num)[:length]
    else:
        return str(num).ljust(length, paddingChar)


def linspace(startVal, stopVal, stepVal):
    values = []
    counter = 0
    newVal = startVal;

    while newVal < stopVal:
        values.append(newVal)
        counter += 1
        newVal = startVal+stepVal*counter;
        #div = round(1/stepVal)
        #newVal = round(newVal * div) / div #avoid floating number precision problems
    return(values)


def extractAllNGrams(words,n):
    assert(n>0)
    startPos = 0
    endPos = len(words)-n
    return [(words[pos:pos+n]) for pos in range(startPos, endPos+1)]


def runFst(sentences, grammarDir, grammarFileName, fstPath):
    assert(os.path.isfile(fstPath))
    writeFile(grammarDir + "/grammarInput.txt", sentences)
    os.chdir(grammarDir)
    #os.system("copy %s %s" % (fstPath, grammarDir))
    cmdStr = "%s -f %s" % (fstPath,grammarFileName)
    print cmdStr
    os.system(cmdStr)


def plotHistogram(list1D, nrBins=100):
    import matplotlib.pyplot as plt
    plt.hist(list1D, nrBins)
    plt.show()


def QCSQueryLabel(qcsQueryLabelPath, modelPath, intputQueriesPath, domainString, variantString, clientIdString):
    qcsQueryLabelCmdPattern = "%s -c %s --variant %s -d %s --clientId %s --input %s"
    modelPath = modelPath.replace('/','\\')
    intputQueriesPath = intputQueriesPath.replace('/','\\')
    cmdStr = qcsQueryLabelCmdPattern % (qcsQueryLabelPath, modelPath, variantString, domainString, clientIdString, intputQueriesPath)
    print cmdStr
    return os.system(cmdStr)


def qasConfigExtractor(qasConfigExtractorPath, qasAllModelsDir, modelString, extractedQasModelDir):
    qasConfigExtractorPath = qasConfigExtractorPath.replace('/','\\')
    qasAllModelsDir = qasAllModelsDir.replace('/','\\')
    extractedQasModelDir = extractedQasModelDir.replace('/','\\')

    cmdStr = "%s %s %s %s" % (qasConfigExtractorPath, qasAllModelsDir, extractedQasModelDir, modelString)
    print cmdStr
    return os.system(cmdStr)






def isAlphabetCharacter(c):
    ordinalValue = ord(c)
    if (ordinalValue>=65 and ordinalValue<=90) or (ordinalValue>=97 and ordinalValue<=122):
        return True
    return False


def isNumberCharacter(c):
    ordinalValue = ord(c)
    if (ordinalValue>=48 and ordinalValue<=57):
        return True
    return False

#def runFstWithAddedSpaces(sentences, grammarDir, grammarFileName, fstPath):
#    #since only match space+dictEntry+space, hence need to add whitespace at begin/end of sentence
#    sentencesWithSpace = ["  "+s+"  " for s in sentences]
#    runFst(sentencesWithSpace, grammarDir, grammarFileName, fstPath)

def textNormalizeQuery(query, textNormalizationLUT):
    textNormalizeQuery = query[:]
    for key,mappedKey in textNormalizationLUT.items():
        textNormalizeQuery = textNormalizeQuery.replace(key,mappedKey)
        #this is a hack until I understand why loading from different text files
        #resulted in differently formatted strings
        #try:
        #textNormalizeQuery = textNormalizeQuery.decode('latin-1').encode('utf-8')
        textNormalizeQuery = textNormalizeQuery.replace(key,mappedKey)
        #textNormalizeQuery = textNormalizeQuery.decode('utf-8').encode('latin-1')
        #except:
        #    pass
    return textNormalizeQuery




#text normalize queries
def textNormalizeQueries(queries, textNormalizationLUT):
    return [textNormalizeQuery(s, textNormalizationLUT) for s in queries]


def combineDictionaries(dictA, dictB):
    dict = dictA
    for key in dictB:
        dict[key] = dictB[key]
    return dict



def runConlleval(conllevalInputFile, conllevalDir, cygwinDir):
    cmdString = "%s/bin/dos2unix %s" % (cygwinDir,conllevalInputFile)
    os.system(cmdString)
    cmdString = "%s/bin/perl.exe %s/conlleval.pl -d '\t' < %s" % (cygwinDir,conllevalDir,conllevalInputFile)
    print "Executing: " + cmdString
    os.system(cmdString)


def inAzure():
    return os.path.isfile(r'azuremod.py')

def isString(var):
    return isinstance(var, basestring)

def isList(var):
    return isinstance(var, list)

def isTuple(var):
    return isinstance(var, tuple)

def rotatePoint(point, angleInDegrees, centerPoint = [0,0]):
    angleInDegrees = - angleInDegrees #to stay conform with how OpenCVs handles rotation which does counter-clockwise rotation
    while angleInDegrees<0:
        angleInDegrees += 360
    theta = angleInDegrees / 180.0 * pi
    ptXNew = cos(theta) * (point[0]-centerPoint[0]) - sin(theta) * (point[1]-centerPoint[1]) + centerPoint[0]
    ptYNew = sin(theta) * (point[0]-centerPoint[0]) + cos(theta) * (point[1]-centerPoint[1]) + centerPoint[1]
    return [ptXNew, ptYNew]

def avg(list1D):
    return sum(list1D, 0.0) / len(list1D)

def pbMax(list1D):
    maxVal = max(list1D)
    indices = [i for i in range(len(list1D)) if list1D[i] == maxVal]
    return maxVal,indices

#this often does not always show the correct size
def showVars(context, maxNrLines = 10**6):
    varsDictItems = context.items()
    varSizes = []
    for varsDictItem in varsDictItems:
        obj = varsDictItem[1]
        if type(obj) is list:
            size = 0
            for listItem in obj:
                try:
                    size += listItem.nbytes
                except:
                    size += sys.getsizeof(listItem)
            #if varsDictItem[0] == 'feats_test':
            #    pdb.set_trace()
        else:
            size = sys.getsizeof(obj)
        varSizes.append(size)
    #varSizes = [sys.getsizeof(obj) for name,obj in varsDictItems]
    dummy, sortOrder = listSort(varSizes, reverseSort = True)
    print "{0:10} | {1:30} | {2:100}".format("SIZE", "TYPE", "NAME")
    print "="*100
    for index in sortOrder[:maxNrLines]:
        print "{0:10} | {1:30} | {2:100}".format(varSizes[index], type(varsDictItems[index][1]), varsDictItems[index][0])
