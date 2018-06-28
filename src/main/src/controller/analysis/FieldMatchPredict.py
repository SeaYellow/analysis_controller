# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:55:27 2018

@author: merit
"""

import pandas as pd
import cx_Oracle
import numpy as np
import pickle
import time


class FieldMatchApply:

    def getUsedData(self, dbConfigMaps):

        tableNames = []
        dataMap = {}
        colsMap = {}
        for configMap in dbConfigMaps:
            tables = configMap['tables']

            user = configMap['user']
            pwd = configMap['password']
            host = configMap['host']
            port = configMap['port']
            db = configMap['database']
            url = host + ':' + port + '/' + db
            conn = cx_Oracle.connect(user, pwd, url)

            for name in tables:
                value = pd.read_sql('select * from ' + name, conn)
                key = name
                dataMap[key] = value
            for key in dataMap:
                curDf = dataMap[key]
                curCols = curDf.columns.tolist()
                colsMap[key] = curCols
            tableNames += tables

        return [dataMap, colsMap, tableNames]

    def contain_chinese(self, check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def getAlphaNumberCount(self, str_value):
        count = [0, 0, 0, 0, 0]
        for ch in str_value:
            if '0' <= ch <= '9':
                count[1] += 1
            if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
                count[0] += 1
            if u'\u4e00' <= ch <= u'\u9fff':
                count[2] += 1
            if ch == '-':
                count[3] += 1
            if ch == '_' or ch == '—':
                count[4] += 1
        return count

    def getType(self, value):

        if ('int' in str(type(value))):
            return 0
        elif ('float' in str(type(value))):
            return 1
        elif ('str' in str(type(value))):
            return 2
        elif ('None' in str(type(value))):
            return 3
        elif ('LOB' in str(type(value)).upper()):
            return 4
        else:
            return 5

    def typeCanCompare(self, t1, t2):
        flag = False
        list1 = [0, 1, 2]
        list2 = [3, 4, 5]
        if ((t1 in list1) and (t2 in list1)):
            flag = True
        elif ((t1 in list1) or (t2 in list1)):
            flag = False
        elif ((t1 in list2) and (t2 in list2)):
            flag = (t1 == t2)
        else:
            flag = False
        return flag

    # 获取最大公共子序列相似度
    def getLCS(self, s1, s2):
        # reload(sys)
        # sys.setdefaultencoding('utf-8')
        len1 = len(s1)
        len2 = len(s2)
        arr = np.zeros([len2 + 1, len1 + 1])

        for p in range(1, len2 + 1):
            lineUnit = s2[p - 1]
            for q in range(1, len1 + 1):
                leftValue = arr[p, q - 1]
                topValue = arr[p - 1, q]
                cornerValue = arr[p - 1, q - 1]

                if lineUnit == s1[q - 1]:
                    cornerValue += 1
                arr[p, q] = np.max([leftValue, topValue, cornerValue])

        commonLen = arr[len2, len1]
        sim = commonLen / min([len1, len2])
        return sim

    def getValueStats(self, df, colName):
        sampleData = []
        tableSize = len(df)
        if (tableSize > 50):
            sampleData = df.sample(n=50)[colName].tolist()
        else:
            sampleData = df[colName].tolist()

        colTypes = [self.getType(e) for e in sampleData]
        colType = 3
        for tp in colTypes:
            if (tp != 3):
                colType = tp
        lens = []
        meanLen = 0
        if (colType == 4):
            meanLen = 100
        else:
            lens = [len(str(elem)) for elem in sampleData]
            meanLen = np.mean(lens)

        chFlag = 0
        alpFlag = 0
        numFlag = 0
        onlyNum = 0
        onlyAlp = 0
        onlyCh = 0
        chNumMix = 0
        alpNum = 0
        numberNum = 0
        chNum = 0
        concatFlag = 0
        subLineFlag = 0
        if (colType == 4):
            chFlag = 1
            chNum = 100
            onlyCh = 1
        else:
            for value in sampleData:
                s = str(value)
                if (self.contain_chinese(s)):
                    chFlag = 1
            nums = [self.getAlphaNumberCount(str(value)) for value in sampleData]
            numArr = np.array(nums)
            numMean = numArr.mean(axis=0)
            alpNum = numMean[0]
            numberNum = numMean[1]
            chNum = numMean[2]
            concatNum = numMean[3]
            subLineNum = numMean[4]
            if (alpNum > 0):
                alpFlag = 1
            if (numberNum > 0):
                numFlag = 1
            if (concatNum > 0):
                concatFlag = 1
            if (subLineNum > 0):
                subLineFlag = 1
            if (meanLen > 0.0 and numberNum / meanLen > 0.95):
                onlyNum = 1
            if (meanLen > 0.0 and alpNum / meanLen > 0.95):
                onlyAlp = 1
            if (meanLen > 0.0 and chNum / meanLen > 0.9 and numberNum / meanLen < 0.1):
                onlyCh = 1
            if (chNum > 0 and numberNum / meanLen > 0.02):
                chNumMix = 1

        noneCount = 0
        hasNone = 0
        for v in sampleData:
            if (v == None):
                noneCount += 1

        if (noneCount > 0):
            hasNone = 1
        vSet = set(sampleData)
        setLen = len(vSet)
        dataLen = len(sampleData)
        noneRatio = noneCount / dataLen
        setRatio = setLen / dataLen
        distinctValues = []
        if (colType != 4):
            distinctValues = [str(elem) for elem in vSet]

        featureMap = {}
        featureMap['colType'] = colType
        featureMap['meanLen'] = meanLen
        featureMap['chFlag'] = chFlag
        featureMap['alpFlag'] = alpFlag
        featureMap['numFlag'] = numFlag
        featureMap['onlyNum'] = onlyNum
        featureMap['onlyAlp'] = onlyAlp
        featureMap['onlyCh'] = onlyCh
        featureMap['chNumMix'] = chNumMix
        featureMap['concatFlag'] = concatFlag
        featureMap['subLineFlag'] = subLineFlag
        featureMap['alpNum'] = alpNum
        featureMap['numberNum'] = numberNum
        featureMap['chNum'] = chNum
        featureMap['hasNone'] = hasNone
        featureMap['noneRatio'] = noneRatio
        featureMap['setRatio'] = setRatio
        featureMap['distinctValues'] = distinctValues

        return featureMap

    def getCompareFeature(self, colName1, colName2, featureMap1, featureMap2):
        colNameSim = self.getLCS(colName1, colName2)

        len1 = featureMap1['meanLen']
        len2 = featureMap2['meanLen']
        maxLen = np.max([len1, len2])
        subLen = abs(len1 - len2)
        lenDif = 0.0
        if (subLen == 0.0):
            lenDif = 0.0
        else:
            lenDif = subLen / maxLen

        hasChSame = 0
        if (featureMap1['chFlag'] == featureMap2['chFlag']):
            hasChSame = 1
        hasAlpSame = 0
        if (featureMap1['alpFlag'] == featureMap2['alpFlag']):
            hasAlpSame = 1
        hasNumSame = 0
        if (featureMap1['numFlag'] == featureMap2['numFlag']):
            hasNumSame = 1

        alpNum1 = featureMap1['alpNum']
        alpNum2 = featureMap2['alpNum']
        maxAlpNum = np.max([alpNum1, alpNum2])
        alpSubNum = abs(alpNum1 - alpNum2)
        alpDif = 0.0
        if (alpSubNum == 0.0):
            alpDif = 0.0
        else:
            alpDif = alpSubNum / maxAlpNum

        numberNum1 = featureMap1['numberNum']
        numberNum2 = featureMap2['numberNum']
        maxNumberNum = np.max([numberNum1, numberNum2])
        numberSubNum = abs(numberNum1 - numberNum2)
        numberDif = 0.0
        if (numberSubNum == 0.0):
            numberDif = 0.0
        else:
            numberDif = numberSubNum / maxNumberNum

        chNum1 = featureMap1['chNum']
        chNum2 = featureMap2['chNum']
        maxChNum = np.max([chNum1, chNum2])
        chSubNum = abs(chNum1 - chNum2)
        chDif = 0.0
        if (chSubNum == 0.0):
            chDif = 0.0
        else:
            chDif = chSubNum / maxChNum

        hasNoneSame = 0
        if (featureMap1['hasNone'] == featureMap2['hasNone']):
            hasNoneSame = 1

        onlyNumSame = 0
        if (featureMap1['onlyNum'] == featureMap2['onlyNum']):
            onlyNumSame = 1
        onlyAlpSame = 0
        if (featureMap1['onlyAlp'] == featureMap2['onlyAlp']):
            onlyAlpSame = 1
        onlyChSame = 0
        if (featureMap1['onlyCh'] == featureMap2['onlyCh']):
            onlyChSame = 1
        chNumMixSame = 0
        if (featureMap1['chNumMix'] == featureMap2['chNumMix']):
            chNumMixSame = 1
        concatSame = 0
        if (featureMap1['concatFlag'] == featureMap2['concatFlag']):
            concatSame = 1
        subLineSame = 0
        if (featureMap1['subLineFlag'] == featureMap2['subLineFlag']):
            subLineSame = 1

        nRatio = 0.0
        noneRatio1 = featureMap1['noneRatio']
        noneRatio2 = featureMap2['noneRatio']
        minR = np.min([noneRatio1, noneRatio2])
        maxR = np.max([noneRatio1, noneRatio2])
        if (maxR == 0.0):
            nRatio = 1.0
        else:
            nRatio = minR / maxR

        sRatio = 0.0
        setRatio1 = featureMap1['setRatio']
        setRatio2 = featureMap2['setRatio']
        minS = np.min([setRatio1, setRatio2])
        maxS = np.max([setRatio1, setRatio2])
        if (maxS == 0.0):
            sRatio = 1.0
        else:
            sRatio = minS / maxS

        valueSim = 0.0
        type1 = featureMap1['colType']
        type2 = featureMap2['colType']
        if (type1 != 4 and type2 != 4):
            distinct1 = featureMap1['distinctValues']
            distinct2 = featureMap2['distinctValues']
            values = [distinct1, distinct2]
            largeInd = 1 if (len(distinct2) > len(distinct1)) else 0
            smallInd = 1 - largeInd
            small = values[smallInd]
            large = values[largeInd]
            maxNum = len(large)
            minNum = np.min([5, len(small)])
            smallValues = small[0:minNum]
            simMat = np.zeros([maxNum, minNum])
            for ii in range(maxNum):
                for jj in range(minNum):
                    simMat[ii, jj] = self.getLCS(large[ii], smallValues[jj])
            valueSim = simMat.max(axis=0).max()
        if (type1 == 4 and type2 == 4):
            valueSim = 1.0

        return [colNameSim, lenDif, valueSim, hasChSame, hasAlpSame, \
                hasNumSame, alpDif, numberDif, hasNoneSame, nRatio, sRatio, \
                chDif, onlyNumSame, onlyAlpSame, onlyChSame, chNumMixSame, concatSame, subLineSame]

    def predictSingle(self, model, usedData):
        dataMap = usedData[0]
        colsMap = usedData[1]
        tableList = usedData[2]

        tableNum = len(tableList)
        colsList = [colsMap[term] for term in tableList]

        resList = []
        for p in range(tableNum):
            curDf = dataMap[tableList[p]]
            curCols = colsList[p]
            for col in curCols:
                curFeaMap = self.getValueStats(curDf, col)
                if (curFeaMap['noneRatio'] > 0.85):
                    continue
                curRes = []
                for t in range(p):
                    curRes.append("")
                curRes.append(col)
                for q in range(p + 1, tableNum):
                    laterTrainList = []
                    laterColList = []
                    laterMapCol = ""
                    laterAllCols = colsList[q]
                    laterDf = dataMap[tableList[q]]
                    for laterCol in laterAllCols:
                        laterFeaMap = self.getValueStats(laterDf, laterCol)
                        canCompare = self.typeCanCompare(curFeaMap['colType'], laterFeaMap['colType'])
                        if (canCompare and laterFeaMap['noneRatio'] < 0.85):
                            curTrainFea = self.getCompareFeature(col, laterCol, curFeaMap, laterFeaMap)
                            laterColList.append(laterCol)
                            laterTrainList.append(curTrainFea)
                    if (len(laterTrainList) > 0):
                        curPreds = model.predict(laterTrainList)
                        predPosInds = []
                        for ind in range(len(curPreds)):
                            if (curPreds[ind] == 1):
                                predPosInds.append(ind)
                        if (len(predPosInds) > 0):
                            probPredData = np.array(laterTrainList)[predPosInds]
                            probs = model.predict_proba(probPredData)
                            probs_pos = probs[:, 1]
                            maxInd = probs_pos.tolist().index(max(probs_pos))
                            colInd = predPosInds[maxInd]
                            laterMapCol = laterColList[colInd]
                            laterAllCols.remove(laterMapCol)
                    curRes.append(laterMapCol)
                noneCount = 0
                for item in curRes:
                    if (item == None):
                        noneCount += 1
                if (len(curRes) - noneCount > 1):
                    resList.append(curRes)
        for term in resList:
            print('len: ' + str(len(term)))
        resDf = pd.DataFrame(resList, columns=tableList)
        print(resDf)
        return str(resDf.to_dict(orient='list'))

    def fieldPred(self, dbConfigMaps, modelPath):
        usedData = self.getUsedData(dbConfigMaps)
        newModel = pickle.load(open(modelPath, 'rb'))
        res = self.predictSingle(newModel, usedData)
        res = res.replace('\'','\"')
        colsMap = usedData[1]
        tableCols = str(colsMap)
        tableCols = tableCols.replace('\'','\"')
        return [res, tableCols]


# -------------------------------------------------
# configMap = {}
# configMap['user'] = 'portaldb'
# configMap['password'] = 'portaldb'
# configMap['host'] = '191.168.7.45'
# configMap['port'] = '1521'
# configMap['database'] = 'orcl'
#
# configMap02 = configMap.copy()
# configMap03 = configMap.copy()
#
# configMap['tables'] = ['T_TX_ZNYC_DZ']
# configMap02['tables'] = ['T_SB_ZNYC_DZ']
# configMap03['tables'] = ['G_SUBS']
#
# dbConfigMaps = [configMap, configMap02, configMap03]
#
# # tableMapPath = 'D:\\data\\xm\\TABLE_MAP.csv'
# modelPath = 'D:\\tmp\\field'
# table1 = 'G_LINE'
# table2 = 'T_TX_ZWYC_DKX'
# table3 = 'T_SB_ZWYC_DKX'
# start = time.clock()
# modeler = FieldMatchApply()
# myRes = modeler.fieldPred(dbConfigMaps, modelPath)
# print(myRes[0])
# print('==================================')
# print(myRes[1])
# end = time.clock()
# print('**********************************')
# print('Running time: %s Seconds' % (end - start))