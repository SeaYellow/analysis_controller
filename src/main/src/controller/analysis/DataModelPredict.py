# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 08:44:45 2018

@author: merit
"""

import pandas as pd
import cx_Oracle
import numpy as np
import pickle
from sklearn.cluster import KMeans
import json
import time


class DataModel:

    def getDataMap(self, config):
        dataMap = {}
        for i in range(len(config)):
            user = config[i]['user']
            pwd = config[i]['password']
            host = config[i]['host']
            port = config[i]['port']
            db = config[i]['database']
            url = host + ':' + port + '/' + db
            conn = cx_Oracle.connect(user, pwd, url)

            for j in range(len(config[i]['tables'])):
                key = config[i]['tables'][j]
                value = pd.read_sql('select * from ' + key, conn)
                if len(value)>0:
                    dataMap[key] = value
        print('读库数据结束...')
        return dataMap

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
            if(len(lens)>0):
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

    def getPredictFeatureDf(self, dataMap):
        colsMap = {}
        for table in dataMap:
            cols = dataMap[table].columns.tolist()
            colsMap[table] = cols

        predData = []
        for tableName in dataMap:
            curDf = dataMap[tableName]
            allCols = curDf.columns.tolist()
            curLen = len(curDf)
            for colName in allCols:
                curFeas = self.getValueStats(curDf, tableName, colName, curLen, colsMap)
                predData.append(curFeas)
        predDf = pd.DataFrame(predData, columns= \
            ['tableName', 'colName', 'valueLen', 'containChn', 'colType', 'isFirstCol', 'withId', 'withNo',
             'similarity', \
             'colFreq', 'isSingleCol', 'colPosGroup', 'otherMaxSim', 'manyNone', 'hasNone', 'setRatio'])
        return predDf

    def predict(self, modelPath, predDf):
       # newModel = pickle.load(open(modelPath+'\\rf.pkl', 'rb'))
        newModel = pickle.load(open(modelPath, 'rb'))
        allCols = predDf.columns.tolist()
        features = allCols[2:len(allCols)]
        X = np.array(predDf[features])
        preds = newModel.predict(X)
        predDf['prediction'] = preds
        newDf = predDf[predDf['prediction'] != 2]
        func = lambda x: ('PK' if (x == 0) else 'FK')
        newDf.loc[:, 'keyType'] = newDf.loc[:, 'prediction'].apply(func)
        resDf = newDf[['tableName', 'colName', 'keyType']]
        return resDf

    def conv_to_json(self, dic, b_name, config):
        ########输入是dataframe
        # table1,table2,id1,id2
        data = dic
        temp = {}
        tab = {}
        tables = []
        rela = {}
        relationship = []
        for i in range(len(b_name)):
            tab = {}
            tab["id"] = i + 1
            tab["tablename"] = b_name[i]
            for j in range(len(config)):
                if b_name[i] in config[j]['tables']:
                    tab['dataSourceId'] = config[j]['dataSourceId']
            tables.append(tab)
        for i in data:
            value = data[i]
            tab = {}
            rela = {}
            i = str(i)
            i = i.lstrip("('").rstrip("')")
            i = i.split(",")
            i[0] = i[0].rstrip("'")
            i[1] = i[1].lstrip(" '")
            rela["sourceCol"] = i[0] + '.' + value[0]
            rela["targetCol"] = i[1] + '.' + value[1]
            relationship.append(rela)

        temp["tables"] = tables
        temp["relationships"] = relationship
        return temp

    def is_relationship1(self, a, b):
        aa = a.drop_duplicates().reset_index()
        del aa['index']
        bb = b.drop_duplicates().reset_index()
        del bb['index']
        m = len(set(aa.iloc[:, 0]) & set(bb.iloc[:, 0]))
        return (round(m / min(len(aa), len(bb)), 2))

    def tz(self, x):
        for i in range(len(x)):
            x_ = x.loc[i, 'x']
            x_type = type(x_)  # 类型
            if 'str' in str(x_type):
                x1 = 1  ##字符型
            elif 'float' in str(x_type):
                x1 = 2  ##浮点型
                if (x_ - int(x_)) == 0:
                    x_ = int(x_)
                    x1 = 3
            elif 'int' in str(x_type):
                x1 = 3  ##整型
            elif 'LOB' in str(x_type):
                x1 = 4  ##整型
            else:
                x1 = 0
            x2 = len(str(x_))  # 长度
            if x1 != 1:
                x_ = str(x_)
            x4 = 0
            x3 = 0
            x6 = 0
            for ch in x_:
                if '0' <= ch <= '9':
                    x4 += 1
                if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
                    x3 += 1
                if '\u4e00' <= ch <= '\u9fa5':
                    x6 += 1
            x5 = x2 - x3 - x4 - x6
            x.loc[i, 'type'] = x1
            x.loc[i, 'len'] = x2
            x.loc[i, 'num'] = x4
            x.loc[i, 'xyz'] = x3
            x.loc[i, 'other'] = x5
            x.loc[i, 'hanzi'] = x6
        return x

    def model(self, dataMap, key, tableNames, config):
        key = key.reset_index()
        del key['index']
        x = pd.DataFrame()  # len(key)
        for i in range(len(key)):
            # print(i)
            table = dataMap[key.loc[i, 'tableName']][key.loc[i, 'colName']]
            sam = min(100, len(table))
            t = pd.DataFrame(table.sample(sam))
            # t=pd.read_sql('select t.'+key.loc[i,'colName']+' from '+ key.loc[i,'tableName']+' SAMPLE (50) t where rownum <=50',conn)
            t.columns = ['x']
            t = t.dropna()
            if len(t) == 0:
                t = pd.DataFrame(table)
                t.columns = ['x']
                t = t.dropna()
            if len(t) > 0:
                t['tableName'] = key.loc[i, 'tableName']
                t['colName'] = key.loc[i, 'colName']
                t = t.reset_index()
                del t['index']
                self.tz(t)
                x = x.append(t)
        x = x.reset_index()
        del x['index']
        clf = KMeans(n_clusters=6)
        y_pred = clf.fit_predict(x.iloc[:, 3:])
        for i in range(len(x)):
            x.loc[i, 'kmeans'] = y_pred[i]
        # x_show=x.copy()
        del x['x']
        x = x.drop_duplicates().reset_index()
        del x['index']
        y = x.iloc[:, :2]
        y['kmeans'] = x['kmeans']
        y = y.drop_duplicates().reset_index()
        del y['index']
        model = pd.DataFrame()
        cluser = y['kmeans']
        cluser = cluser.drop_duplicates().reset_index()
        del cluser['index']
        for i in range(len(cluser)):
            cluser_i = y[y['kmeans'] == cluser.loc[i, 'kmeans']].reset_index()
            del cluser_i['index']
            for j in range(len(cluser_i)):
                yw = key[key['tableName'] == cluser_i.loc[j, 'tableName']][
                    key[key['tableName'] == cluser_i.loc[j, 'tableName']]['colName'] == cluser_i.loc[j, 'colName']][
                    'keyType'].reset_index()
                cluser_i.loc[j, 'keyType'] = yw.loc[0, 'keyType']
                key_type = cluser_i['keyType'].drop_duplicates().reset_index()
            if len(key_type) > 1:
                p_key = cluser_i[cluser_i['keyType'] == 'PK'].reset_index()
                del p_key['index']
                f_key = cluser_i[cluser_i['keyType'] == 'FK'].reset_index()
                del f_key['index']
                for k in range(len(p_key)):
                    a = dataMap[p_key.loc[k, 'tableName']][p_key.loc[k, 'colName']]
                    for h in range(len(f_key)):
                        if p_key.loc[k, 'tableName'] != f_key.loc[h, 'tableName']:
                            b = dataMap[f_key.loc[h, 'tableName']][f_key.loc[h, 'colName']]
                            rela = self.is_relationship1(a, b)
                            if rela > 0.65:
                                re = pd.DataFrame()
                                re.loc[0, 'table1'] = p_key.loc[k, 'tableName']
                                re.loc[0, 'table2'] = f_key.loc[h, 'tableName']
                                re.loc[0, 'id1'] = p_key.loc[k, 'colName']
                                re.loc[0, 'id2'] = f_key.loc[h, 'colName']
                                model = model.append(re)
        print(time.localtime(time.time()))
        if len(model) > 0:
            model = model.drop_duplicates()
            # model.to_csv('F:\\2018\\2月\\2.26\\model3.csv')

        model = model.reset_index()
        del model['index']
        model_result = {}
        for i in range(len(model)):
            model_result[model.loc[i, 'table1'], model.loc[i, 'table2']] = (model.loc[i, 'id1'], model.loc[i, 'id2'], 1)
        model_result = self.conv_to_json(model_result, tableNames, config)
        return model_result

    def runProcess(self, dbConfigMap, tableNames, modelPath):
        dataMap = self.getDataMap(dbConfigMap)
        predDf = self.getPredictFeatureDf(dataMap)
        resDf = self.predict(modelPath, predDf)
        jsObj = json.dumps(self.model(dataMap, resDf, tableNames, dbConfigMap))
        #res = eval(jsObj)
        return jsObj
'''
configMap = {}
configMap['user'] = 'C##meritdata'
configMap['password'] = 'meritdata8991'
configMap['host'] = '191.168.6.68'
configMap['port'] = '1521'
configMap['database'] = 'orcl'
configMap['tables'] = ['ARC_S_APP_VAT', 'ARC_S_DESIGN_EXAMINE', 'ARC_S_PRC_TACTIC_SCHEME', 'ARC_S_PSP_DESIGN_VERI', 'ARC_S_PSP_DESIGN', 'ARC_S_PS_SCHEME', 'ARC_S_PRJ_INSPECT', 'ARC_S_PS_SCHEME_DRAWING', 'ARC_S_PSP_CONSTRUCT', 'ARC_S_APP_PAYMENT_RELA', 'ARC_S_MI_RELA_SCHEME', 'ARC_S_APP', 'ARC_S_APP_DATA', 'ARC_S_APP_REPLY', 'ARC_S_BILL_RELA_SCHEME', 'ARC_S_CONSPRC_SCHEME', 'ARC_S_INVESTIGATE', 'ARC_S_METER_SCHEME', 'ARC_S_SND_CIRCUIT_SCHEME', 'ARC_S_MID_CHECK', 'ARC_S_IT_SCHEME']
configMap['dataSourceId']=5
configMap2 = {}
configMap2['user'] = 'C##meritdata'
configMap2['password'] = 'meritdata8991'
configMap2['host'] = '191.168.6.68'
configMap2['port'] = '1521'
configMap2['database'] = 'orcl'
configMap2['tables'] = ['ARC_S_PRJ_ACCEPT', 'ARC_S_APP_CERT', 'ARC_S_APP_BANK_ACCT', 'ARC_S_ELEC_DEV_SCHEME', 'ARC_S_MP_METER_RELA_SCHM', 'ARC_S_APP_NATURAL_INFO', 'ARC_S_PS_CHG_SCHEME', 'ARC_S_APP_ELEC_ADDR', 'ARC_S_MP_SCHEME', 'ARC_S_APP_CONTACT', 'ARC_S_APP_ACCT']
configMap2['dataSourceId']=6
tableNames= ['ARC_S_APP_VAT', 'ARC_S_DESIGN_EXAMINE', 'ARC_S_PRC_TACTIC_SCHEME', 'ARC_S_PSP_DESIGN_VERI', 'ARC_S_PSP_DESIGN', 'ARC_S_PS_SCHEME', 'ARC_S_PRJ_INSPECT', 'ARC_S_PS_SCHEME_DRAWING', 'ARC_S_PSP_CONSTRUCT', 'ARC_S_APP_PAYMENT_RELA', 'ARC_S_MI_RELA_SCHEME', 'ARC_S_APP', 'ARC_S_APP_DATA', 'ARC_S_APP_REPLY', 'ARC_S_BILL_RELA_SCHEME', 'ARC_S_CONSPRC_SCHEME', 'ARC_S_INVESTIGATE', 'ARC_S_METER_SCHEME', 'ARC_S_SND_CIRCUIT_SCHEME', 'ARC_S_MID_CHECK', 'ARC_S_IT_SCHEME', 'ARC_S_PRJ_ACCEPT', 'ARC_S_APP_CERT', 'ARC_S_APP_BANK_ACCT', 'ARC_S_ELEC_DEV_SCHEME', 'ARC_S_MP_METER_RELA_SCHM', 'ARC_S_APP_NATURAL_INFO', 'ARC_S_PS_CHG_SCHEME', 'ARC_S_APP_ELEC_ADDR', 'ARC_S_MP_SCHEME', 'ARC_S_APP_CONTACT', 'ARC_S_APP_ACCT']
config=[configMap,configMap2]


#print(time.localtime(time.time())) 
#configMap = {}
#configMap['user'] = 'portaldb'
#configMap['password'] = 'portaldb'
#configMap['host'] = '191.168.7.45'
#configMap['port'] = '1521'
#configMap['database'] = 'orcl'
#b_name=['G_SUBS','G_LINE','T_RZ_BG_SUM_DX']
modelPath = 'F:\\2018\\2月\\2.26'

data_model = DataModel()
print('**************************************')
myJson = data_model.runProcess(config, tableNames, modelPath)
print(myJson)

'''
# =====================================================
# print(time.localtime(time.time()))
# configMap = {}
# configMap['user'] = 'caiwu'
# configMap['password'] = 'caiwu'
# configMap['host'] = '191.168.6.68'
# configMap['port'] = '1521'
# configMap['database'] = 'orcl'
# b_name=['TRDATADETAIL2407','TRDATADETAIL2418','TRDATADETAIL3878','TRDATADETAIL4931','TRDATADETAIL4934','TRDATADETAIL5060','TRDATADETAIL5061','TRDATADETAIL5063','T_RZ_BG_SUM_DX','T_ZJ_BALCHANGE','T_ZJ_BALCHANGEMX','XT10031YWBILL','XT167YWBILL',]
# modelPath = 'F:\\2018\\2月\\2.26'
#
# data_model = DataModel()
# print('**************************************')
# myJson = data_model.runProcess(configMap, b_name, modelPath)
# print(myJson)

# configMap = {}
# configMap['user'] = 'yangwen2'
# configMap['password'] = 'yangwen2'
# configMap['host'] = '191.168.6.68'
# configMap['port'] = '1521'
# configMap['database'] = 'orcl'
# configMap['tables'] = ['ARC_S_APP', 'ARC_S_APP_ACCT']
#
# configMap2 = {}
# configMap2['user'] = 'yangwen2'
# configMap2['password'] = 'yangwen2'
# configMap2['host'] = '191.168.6.68'
# configMap2['port'] = '1521'
# configMap2['database'] = 'orcl'
# configMap2['tables'] = ['ARC_S_APP_CUST_ADDR', 'ARC_S_APP_DATA', 'ARC_S_APP_ELEC_ADDR', 'ARC_S_APP_NATURAL_INFO']
#
# tableNames = ['ARC_S_APP', 'ARC_S_APP_ACCT', 'ARC_S_APP_CUST_ADDR', 'ARC_S_APP_DATA', 'ARC_S_APP_ELEC_ADDR',
#               'ARC_S_APP_NATURAL_INFO']
#
# config = [configMap, configMap2]

# print(time.localtime(time.time()))
# configMap = {}
# configMap['user'] = 'portaldb'
# configMap['password'] = 'portaldb'
# configMap['host'] = '191.168.7.45'
# configMap['port'] = '1521'
# configMap['database'] = 'orcl'
# b_name=['G_SUBS','G_LINE','T_RZ_BG_SUM_DX']
# modelPath = 'F:\\2018\\2月\\2.26'
#
# data_model = DataModel()
# print('**************************************')
# myJson = data_model.runProcess(config, tableNames, modelPath)
# print(myJson)

