# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:22:35 2018

@author: merit
"""

import pandas as pd
import cx_Oracle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import pickle
import time
from sklearn import metrics


class FieldMatchTrain:
    def getUsedData(self, dbConfigMap, orientTrainDataPath):
        df = pd.read_csv(orientTrainDataPath, header=0, sep=',')
        tableNames1 = df['TABLE1'].tolist()
        tableNames2 = df['TABLE2'].tolist()
        tableSet = set(tableNames1 + tableNames2)
        tableNames = [t_name for t_name in tableSet]

        user = dbConfigMap['user']
        pwd = dbConfigMap['password']
        host = dbConfigMap['host']
        port = dbConfigMap['port']
        db = dbConfigMap['database']
        url = host + ':' + port + '/' + db
        conn = cx_Oracle.connect(user, pwd, url)

        dataMap = {}

        for name in tableNames:
            value = pd.read_sql('select * from ' + name, conn)
            key = name
            dataMap[key] = value
        print('读库数据结束...')

        colsMap = {}
        for key in dataMap:
            curDf = dataMap[key]
            curCols = curDf.columns.tolist()
            colsMap[key] = curCols

        return [df, dataMap, colsMap]

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

    def getTrainData(self, usedData):
        df = usedData[0]
        dataMap = usedData[1]
        trainData = []
        tableNum = len(df)

        for ind in range(tableNum):
            curLine = df.ix[ind]
            tableName1 = curLine['TABLE1']
            tableName2 = curLine['TABLE2']
            col1 = curLine['COL1']
            col2 = curLine['COL2']
            flag = curLine['IS_CONSISTENT']

            label = 1 if (flag == 'YES') else 0
            feaMap1 = self.getValueStats(dataMap[tableName1], col1)
            feaMap2 = self.getValueStats(dataMap[tableName2], col2)
            compareFea = self.getCompareFeature(col1, col2, feaMap1, feaMap2)
            compareFea.append(label)
            trainData.append(compareFea)

        trainDf = pd.DataFrame(trainData, columns= \
            ['colNameSim', 'lenDif', 'valueSim', 'hasChSame', 'hasAlpSame', 'hasNumSame', \
             'alpDif', 'numberDif', 'hasNoneSame', 'noneRatio', 'setRatio', 'chDif', 'onlyNumSame', \
             'onlyAlpSame', 'onlyChSame', 'chNumMixSame', 'concatSame', 'subLineSame', 'label'])
        return trainDf

    def trainModel(self, df, savePath):
        allCols = df.columns.tolist()

        y = df['label'].tolist()
        features = allCols[0:len(allCols) - 1]
        X = np.array(df[features])

        rf = RandomForestClassifier(n_estimators=30,
                                    criterion="entropy",
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.,
                                    max_features=None,
                                    max_leaf_nodes=None,
                                    bootstrap=True,
                                    oob_score=False,
                                    n_jobs=1,
                                    random_state=None,
                                    verbose=0,
                                    warm_start=False,
                                    class_weight={1: 0.6, 0: 0.4})
        model = rf.fit(X, y)
        preds = model.predict(X)
        print('*******************************')
        resDf = pd.DataFrame()
        resDf['label'] = y
        resDf['preds'] = preds
        print(resDf)
        confuseMat = metrics.confusion_matrix(y, preds)
        acc = metrics.accuracy_score(y, preds)
        recall = metrics.recall_score(y, preds)

        print('==================================')
        print('confuseMatrix:')
        print(confuseMat)
        print('准确率，召回率：')
        print(str(acc) + '---' + str(recall))
        with open(savePath, 'wb') as f:
            pickle.dump(model, f)
        return model

    def trainAndSave(self, trainFilePath, modelSavePath, dbConfigMap):
        usedData = self.getUsedData(dbConfigMap, trainFilePath)
        trainData = self.getTrainData(usedData)
        self.trainModel(trainData, modelSavePath)

        # -------------------------------------------------------------
        #
# configMap = {}
# configMap['user'] = 'portaldb'
# configMap['password'] = 'portaldb'
# configMap['host'] = '191.168.2.43'
# configMap['port'] = '1521'
# configMap['database'] = 'oracle11'
# start = time.clock()
# filePath = '/home/huanghai/huanghai/analysisController/data/FIELD_MAP_DATA2.csv'
# savePath = '/home/huanghai/huanghai/analysisController/data/field_model.pkl'
# trainer = FieldMatchTrain()
# trainer.trainAndSave(filePath, savePath, configMap)
# print('训练及保存模型完成！')
# end = time.clock()
# print('运行用时：%s 秒'%(end-start))
