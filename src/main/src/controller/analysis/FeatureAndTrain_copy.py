# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:26:08 2018

@author: merit
"""

import pandas as pd
import cx_Oracle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


class TableFeatureAndTrain:
    def getUsedData(self, dbConfigMap, orientTrainDataPath):
        df = pd.read_csv(orientTrainDataPath, header=0, sep=',')
        tableNames = df['表名'].tolist()

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

    def getValueStats(self, df, tableName, colName, size, colsMap):
        data = []
        allCols = df.columns.tolist()
        if (size > 20):
            data = df.sample(n=20)[colName].tolist()
        else:
            data = df[colName].tolist()

        colTypes = [self.getType(e) for e in data]
        colType = 3
        for tp in colTypes:
            if (tp != 3):
                colType = tp
        lens = []
        meanLen = 0
        if (colType == 4):
            meanLen = 100
        else:
            lens = [len(str(elem)) for elem in data]
            meanLen = np.mean(lens)

        flag = 0
        if (colType == 4):
            flag = 1
        else:
            for value in data:
                s = str(value)
                if (self.contain_chinese(s)):
                    flag = 1

        ind = allCols.index(colName)
        pos = 1
        if (ind > 0):
            pos = 0

        interval = 0
        colNum = len(allCols)
        if (colNum * 1.0 / 3 <= ind and ind < colNum * 2.0 / 3):
            interval = 1
        elif (ind >= colNum * 2.0 / 3):
            interval = 2

        withId = 0
        withNo = 0
        if (colType != 4):
            if (colName.upper().startswith('ID') or colName.upper().endswith('ID')):
                withId = 1
            if (colName.upper().startswith('NO') or colName.upper().endswith('NO')):
                withNo = 1

        sim = self.getLCS(tableName, colName)

        freq = 0
        for key in colsMap:
            curList = colsMap[key]
            for col in curList:
                if (col == colName):
                    freq += 1

        alone = 1
        if (freq > 1):
            alone = 0

        otherSims = []
        for key in colsMap:
            if (key != tableName):
                curSim = self.getLCS(key, colName)
                otherSims.append(curSim)
        otherMaxSim = np.max(otherSims)
        noneCount = 0
        manyNone = 0
        hasNone = 0
        for v in data[0:10]:
            if (v == None):
                noneCount += 1
        if (noneCount > 7):
            manyNone = 1
        if (noneCount > 0):
            hasNone = 1
        setLen = len(set(data))
        dataLen = len(data)
        setRatio = setLen / dataLen

        return [tableName, colName, meanLen, flag, colType, pos, withId, withNo, \
                sim, freq, alone, interval, otherMaxSim, manyNone, hasNone, setRatio]

    def getTrainData(self, usedData):
        df = usedData[0]
        dataMap = usedData[1]
        colsMap = usedData[2]
        trainData = []
        tableNum = len(df)

        for ind in range(tableNum):
            curLine = df.ix[ind]
            curTableName = curLine['表名']
            curPK = curLine['主键(业务)'].strip().split(';')
            curPK = [elem.strip() for elem in curPK]
            curFK = []
            if (str(curLine['外键(业务)']) != 'nan'):
                curFK = curLine['外键(业务)'].strip().split(';')
                curFK = [elem.strip() for elem in curFK]
            curOther = curLine['其它字段'].strip().split(';')
            curOther = [elem.strip() for elem in curOther]

            curDf = dataMap[curTableName]
            curSize = len(curDf)

            for pk in curPK:
                curRes = self.getValueStats(curDf, curTableName, pk, curSize, colsMap)
                curRes.append(0)
                trainData.append(curRes)
            for fk in curFK:
                curRes = self.getValueStats(curDf, curTableName, fk, curSize, colsMap)
                curRes.append(1)
                trainData.append(curRes)
            for ot in curOther:
                curRes = self.getValueStats(curDf, curTableName, ot, curSize, colsMap)
                curRes.append(2)
                trainData.append(curRes)
        return pd.DataFrame(trainData,
                            columns=['tableName', 'colName', 'valueLen', 'containChn', 'colType', 'isFirstCol',
                                     'withId', 'withNo', 'similarity', 'colFreq', 'isSingleCol', 'colPosGroup',
                                     'otherMaxSim', 'manyNone', 'hasNone', 'setRatio', 'class'])

    def trainModel(self, df, savePath):
        allCols = df.columns.tolist()

        y = df['class'].tolist()
        features = allCols[2:len(allCols) - 1]
        X = np.array(df[features])

        rf = RandomForestClassifier()
        model = rf.fit(X, y)
        with open(savePath, 'wb') as f:
            pickle.dump(model, f)
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=')
        print(model.predict_proba(X))
        return model

    def runFeatureTrain(self, trainFilePath, modelSavePath, dbConfigMap):
        usedData = self.getUsedData(dbConfigMap, trainFilePath)
        trainData = self.getTrainData(usedData)
        self.trainModel(trainData, modelSavePath)

# =========================================================
'''
configMap = {}
configMap['user'] = 'C##meritdata'
configMap['password'] = 'meritdata8991'
configMap['host'] = '191.168.6.68'
configMap['port'] = '1521'
configMap['database'] = 'orcl'
#
filePath = '/home/prog/algorithm/analysisController/data/primary_foreign_key.csv'
savePath = '/home/prog/algorithm/analysisController/data/rf.pkl'
trainer = TableFeatureAndTrain()
print('开始训练模型')
trainer.runFeatureTrain(filePath, savePath, configMap)
print('训练及保存模型完成！')
'''
