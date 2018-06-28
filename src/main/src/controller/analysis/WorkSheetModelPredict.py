# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:31:27 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:29:07 2018

@author: wuxf
"""

import queue
import cx_Oracle
import pandas as pd
import numpy as np
import time
import json


class WorkSheetModelPredict:
    """
    工单分布分析类。以一张表为源表，其ID列为源列，对其中每一条记录(以id标记)，
    查询与其直接或间接关联的表中是否存在关联记录。存在则标记1，不存在则标记0.
    由此，对于源表中的每一条记录，计算出一个向量，其长度为输入表的个数，每个
    分量为上述标记。
    对于源表中的所有记录，求出上述向量，然后按唯一值分组，最后求出百分比，
    作为分布描述。
    """

    ##根据提供的数据库地址和表明，将表存在datamap中
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
                if len(value) > 0:
                    dataMap[key] = value
        print('读库数据结束...')
        return dataMap
        # 根据存储的每个表的前置pre结果，获取从源表到目标表的连通路径。

    # 这里每个表的位置，由其name在tableNames中的下标表示。
    def getPath(self, pres, end, path):
        if pres[end] == -1:
            path.append(end)
            return path
        else:
            path.append(end)
            return self.getPath(pres, pres[end], path)

    # 分析start到end的连通性，然后获取其连通路径。
    # 若连通，则调用getPath()获取其路径；否则返回[]。
    def getRoad(self, mat, start, end):
        n = len(mat)
        visited = [0 for x in range(n)]
        pres = [0 for x in range(n)]
        flag = 0
        que = queue.Queue()
        que.put(start)
        visited[start] = 1
        pres[start] = -1

        while (que.empty() == False):
            front = que.get()
            for i in range(n):
                if (mat[front][i] == 1 and visited[i] == 0):
                    visited[i] = 1
                    que.put(i)
                    pres[i] = front
                    if (i == end):
                        flag = 1
                        break
        if (flag == 0):
            return []
        else:
            res = self.getPath(pres, end, [])
            res.reverse()
            return res

    # 以某个固定的start为起点，计算出其到所有点的连通路径
    def getAllRoad(self, mat, start):
        nodeNum = len(mat)
        res = []
        for i in range(nodeNum):
            curRoad = self.getRoad(mat, start, i)
            res.append(curRoad)
        return res

    def getRelaName(self, tableNames, relaMap):
        tableNum = len(tableNames)
        relaMat = [[0 for p in range(tableNum)] for q in range(tableNum)]
        single = []
        relaname = {}
        relation = {}
        k = 0
        for key in relaMap:
            t1 = key[0]
            t2 = key[1]
            ind1 = tableNames.index(t1)
            ind2 = tableNames.index(t2)
            relaMat[ind1][ind2] = 1
            relaMat[ind2][ind1] = 1
        relaMat1 = pd.DataFrame(relaMat)
        for i in range(len(relaMat1)):
            if (np.mean(relaMat1.loc[:, i]) == 0):
                single.append(tableNames[i])
            else:
                x = relaMat1.loc[i + 1:, i][relaMat1.loc[i + 1:, i] == 1]
                index = list(x.index)
                index.append(i)
                relation[k] = index
                k = k + 1
        for t in range(k):
            for l in range(k):
                if (t != l):
                    a = relation[t]
                    b = relation[l]
                    if (len(list(set(a).intersection(set(b)))) > 0):
                        for i in range(len(relation[l])):
                            relation[t].append(relation[l][i])
                        relation[t] = list(set(relation[t]))
                        # ralation.pop(l)
                        relation[l] = []
        for i in range(k):
            if relation[i] == []:
                relation.pop(i)
        n = len(relation)
        for i in relation.keys():
            relaname[i] = []
            for y in relation[i]:
                relaname[i].append(tableNames[y])
        for j in range(len(single)):
            relaname[single[j]] = single[j]
        return relaname
        # 由表名列表(能获取顺序信息)及表间相邻关系，获取邻接矩阵

    def getRalationMat(self, tableNames, relaMap):
        tableNum = len(tableNames)
        relaMat = [[0 for p in range(tableNum)] for q in range(tableNum)]
        for key in relaMap:
            t1 = key[0]
            t2 = key[1]
            if t1 in tableNames:
                if t2 in tableNames:
                    ind1 = tableNames.index(t1)
                    ind2 = tableNames.index(t2)
                    relaMat[ind1][ind2] = 1
                    relaMat[ind2][ind1] = 1
        return relaMat

    # 给定前置表及若干记录，分析其直接关联(相邻)的表中是否存在关联的记录，并返回之
    def getRecordValues(self, df1, valueCol1, values1, conCol1, df2, conCol2):
        newDf1 = df1[df1[valueCol1].isin(values1)]
        conValues1 = newDf1[conCol1].tolist()
        newDf2 = df2[df2[conCol2].isin(conValues1)]
        return newDf2[conCol2].tolist()

    # 由表间关系的dict，给定两个表名，获取其分别的关联字段
    def getConCols(self, relaMap, tableName1, tableName2):
        col1 = ""
        col2 = ""
        if (tableName1, tableName2) in relaMap:
            cols = relaMap[tableName1, tableName2]
            col1 = cols[0]
            col2 = cols[1]
        else:
            cols = relaMap[tableName2, tableName1]
            col1 = cols[1]
            col2 = cols[0]
        return [col1, col2]

    # 从源表开始，判断某一给定路径下的终点表中，是否存在与源表关联的记录
    def hasRecord(self, dataMap, relaMap, tableNames, road, orientCol, orientId):
        roadLen = len(road)
        tableName1 = tableNames[road[0]]
        tableName2 = tableNames[road[1]]
        [col1, col2] = self.getConCols(relaMap, tableName1, tableName2)
        df1 = dataMap[tableName1]
        df2 = dataMap[tableName2]
        values = self.getRecordValues(df1, orientCol, [orientId], col1, df2, col2)
        if (roadLen == 2):
            if (len(values) == 0):
                return 0
            else:
                return 1
        else:
            for i in range(1, roadLen - 1):
                tableName1 = tableNames[road[i]]
                tableName2 = tableNames[road[i + 1]]
                orientCol = col2
                [col1, col2] = self.getConCols(relaMap, tableName1, tableName2)
                df1 = dataMap[tableName1]
                df2 = dataMap[tableName2]
                values = self.getRecordValues(df1, orientCol, values, col1, df2, col2)
                if (len(values) == 0):
                    return 0
            return 1

    # 综合计算标记向量
    def getVector(self, dataMap, tableNames, relaMap, relaMat, orientTable, orientCol, orientId):
        start = tableNames.index(orientTable)
        tableNum = len(tableNames)
        roads = self.getAllRoad(relaMat, start)
        vector = []
        for i in range(tableNum):
            if (i == start):
                vector.append(1)
            elif (len(roads[i]) == 0):
                vector.append(0)
            else:
                curFlag = self.hasRecord(dataMap, relaMap, tableNames, roads[i], orientCol, orientId)
                vector.append(curFlag)
        return vector

    # 计算向量模式分布
    def getModePercents(self, vectors):
        distinctList = []
        counts = []
        for term in vectors:
            if (term not in distinctList):
                distinctList.append(term)
                counts.append(1)
            else:
                ind = distinctList.index(term)
                counts[ind] += 1
        totalNum = len(vectors)
        res = []
        for x in range(len(distinctList)):
            curPercent = round(counts[x] * 100.0 / totalNum, 4)
            res.append((distinctList[x], counts[x], curPercent))
        return res

    # 将上述各个类内串起，形成一个统一的对外方法接口
    # 使用该类的功能时，只需调用此方法即可
    def getDistribution(self, dataMap, relaMap):
        tableNames = []
        for i in dataMap.keys():
            tableNames.append(i)
        relaname = self.getRelaName(tableNames, relaMap)
        df_Vector = pd.DataFrame()
        for i in relaname.keys():
            if type(relaname[i]) != str:
                relaMat = self.getRalationMat(relaname[i], relaMap)
                sumrela = 0
                orientTable = ''
                for j in range(len(relaMat)):
                    if np.sum(relaMat[j]) > sumrela:
                        orientTable = relaname[i][j]
                orientDf = dataMap[orientTable]
                orientCol = self.orientCol(relaMap, orientTable)
                ids = orientDf[orientCol].tolist()
                vectorRes = []
                for id in ids:
                    singleVec = self.getVector(dataMap, relaname[i], relaMap, relaMat, orientTable, orientCol, id)
                    vectorRes.append(singleVec)
                df_vectorRes = pd.DataFrame(vectorRes)
                df_vectorRes.columns = relaname[i]
                df_Vector = df_Vector.append(df_vectorRes)
            if type(relaname[i]) == str:
                vectorRes = []
                for k in range(len(dataMap[relaname[i]])):
                    vectorRes.append(1)
                df_vectorRes = pd.DataFrame(vectorRes)
                df_vectorRes.columns = [relaname[i]]
                df_Vector = df_Vector.append(df_vectorRes)
        df_Vector = df_Vector.reset_index()
        del df_Vector['index']
        df_Vector = df_Vector.fillna(0)
        vectorRes = np.array(df_Vector)
        vectorRes = vectorRes.tolist()
        distribution = self.getModePercents(vectorRes)
        table_Names = (df_Vector.columns).tolist()
        return (distribution, table_Names)

    def convert_val_fmt(self, df, tablename):
        val_fmt = tablename + '_seq.nextval'
        for i in range(len(df.columns)):
            val_fmt += "," + ":" + str(i + 1)
        return '(' + val_fmt[0:] + ')'

    def values_convert(self, df):
        ll = [list(x) for x in df.values]
        result_list = []
        for i, v in enumerate(ll):
            result_list.append([str(a) for a in v])
        tuple_list = [tuple(x) for x in result_list]
        return tuple_list

    def insert_df2oracle(self, df, tablename, cursor):
        str_head = "insert into "
        str_values = " values "
        a = ['ID']
        b = list(df.columns)
        a.extend(b)
        str_columns = str(tuple(a)).replace('\'', '')
        col_format = self.convert_val_fmt(df, tablename)
        str_insert_sql = str_head + tablename + " " + str_columns + str_values + col_format
        tuple_list = self.values_convert(df)
        cursor.executemany(str_insert_sql, tuple_list)

    def work_type(self, FLOW_ID, result, configMap1):
        x = ['WORK_TYPE', 'WORK_NUMBER', 'TYPE_PROPORTION', 'FLOW_ID', 'DATA']
        work_types = pd.DataFrame(columns=x)
        for i in range(len(result[0])):
            work_types.loc[i, 'WORK_TYPE'] = i
            work_types.loc[i, 'WORK_NUMBER'] = result[0][i][1]
            work_types.loc[i, 'TYPE_PROPORTION'] = result[0][i][2]
        data = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        work_types['FLOW_ID'] = FLOW_ID  ####传进来的
        work_types['DATA'] = data
        user = configMap1['user']
        pwd = configMap1['password']
        host = configMap1['host']
        port = configMap1['port']
        db = configMap1['database']
        url = host + ':' + port + '/' + db
        conn = cx_Oracle.connect(user, pwd, url)
        cursor = conn.cursor()
        self.insert_df2oracle(work_types, "work_types", cursor)
        conn.commit()
        return 0

    def work_vector(self, result):
        work_vector = pd.DataFrame(columns=result[1])
        for i in range(len(result[0])):
            for j in range(len(result[1])):
                work_vector.loc[i, result[1][j]] = int(result[0][i][0][j])
                work_vector.loc[i, 'WORK_TYPE'] = i
                vector = []
                for x in range(len(work_vector)):
                    vector.append(work_vector.iloc[x].to_dict())
                vector = json.dumps(vector)
        return vector

    def orientCol(self, relaMap, orientTable):
        table = relaMap.keys()
        table = list(table)
        for i in range(len(table)):
            if table[i][0] == orientTable:
                orientCol = relaMap[table[i]][0]
                break
            elif table[i][1] == orientTable:
                orientCol = relaMap[table[i]][1]
                break
        return orientCol

    def runProcess(self, flowId, params, relaMap, configMap1):
        dataMap = self.getDataMap(params)
        analysisResult = self.getDistribution(dataMap, relaMap)
        print('finish getDistribution()')
        self.work_type(flowId, analysisResult, configMap1)
        print('finish work_type()')
        return self.work_vector(analysisResult)

# relaMap=myJson[1]
# configMap = {}
# configMap['user'] = 'C##meritdata'
# configMap['password'] = 'meritdata8991'
# configMap['host'] = '191.168.6.68'
# configMap['port'] = '1521'
# configMap['database'] = 'orcl'
# configMap['tables'] = ['ARC_S_APP_VAT', 'ARC_S_DESIGN_EXAMINE', 'ARC_S_PRC_TACTIC_SCHEME', 'ARC_S_PSP_DESIGN_VERI', 'ARC_S_PSP_DESIGN', 'ARC_S_PS_SCHEME', 'ARC_S_PRJ_INSPECT', 'ARC_S_PS_SCHEME_DRAWING', 'ARC_S_PSP_CONSTRUCT', 'ARC_S_APP_PAYMENT_RELA', 'ARC_S_MI_RELA_SCHEME', 'ARC_S_APP', 'ARC_S_APP_DATA', 'ARC_S_APP_REPLY', 'ARC_S_BILL_RELA_SCHEME', 'ARC_S_CONSPRC_SCHEME', 'ARC_S_INVESTIGATE', 'ARC_S_METER_SCHEME', 'ARC_S_SND_CIRCUIT_SCHEME', 'ARC_S_MID_CHECK', 'ARC_S_IT_SCHEME']
# configMap['dataSourceId']=5
# configMap2 = {}
# configMap2['user'] = 'C##meritdata'
# configMap2['password'] = 'meritdata8991'
# configMap2['host'] = '191.168.6.68'
# configMap2['port'] = '1521'
# configMap2['database'] = 'orcl'
# configMap2['tables'] = ['ARC_S_PRJ_ACCEPT', 'ARC_S_APP_CERT', 'ARC_S_APP_BANK_ACCT', 'ARC_S_ELEC_DEV_SCHEME', 'ARC_S_MP_METER_RELA_SCHM', 'ARC_S_APP_NATURAL_INFO', 'ARC_S_PS_CHG_SCHEME', 'ARC_S_APP_ELEC_ADDR', 'ARC_S_MP_SCHEME', 'ARC_S_APP_CONTACT', 'ARC_S_APP_ACCT']
# configMap2['dataSourceId']=6
# tableNames= ['ARC_S_APP_VAT', 'ARC_S_DESIGN_EXAMINE', 'ARC_S_PRC_TACTIC_SCHEME', 'ARC_S_PSP_DESIGN_VERI', 'ARC_S_PSP_DESIGN', 'ARC_S_PS_SCHEME', 'ARC_S_PRJ_INSPECT', 'ARC_S_PS_SCHEME_DRAWING', 'ARC_S_PSP_CONSTRUCT', 'ARC_S_APP_PAYMENT_RELA', 'ARC_S_MI_RELA_SCHEME', 'ARC_S_APP', 'ARC_S_APP_DATA', 'ARC_S_APP_REPLY', 'ARC_S_BILL_RELA_SCHEME', 'ARC_S_CONSPRC_SCHEME', 'ARC_S_INVESTIGATE', 'ARC_S_METER_SCHEME', 'ARC_S_SND_CIRCUIT_SCHEME', 'ARC_S_MID_CHECK', 'ARC_S_IT_SCHEME', 'ARC_S_PRJ_ACCEPT', 'ARC_S_APP_CERT', 'ARC_S_APP_BANK_ACCT', 'ARC_S_ELEC_DEV_SCHEME', 'ARC_S_MP_METER_RELA_SCHM', 'ARC_S_APP_NATURAL_INFO', 'ARC_S_PS_CHG_SCHEME', 'ARC_S_APP_ELEC_ADDR', 'ARC_S_MP_SCHEME', 'ARC_S_APP_CONTACT', 'ARC_S_APP_ACCT']
# config=[configMap,configMap2]
#
# configMap1 = {}
# configMap1['user'] = 'C##meritdata'
# configMap1['password'] = 'meritdata8991'
# configMap1['host'] = '191.168.6.68'
# configMap1['port'] = '1521'
# configMap1['database'] = 'orcl'
#
####需要设置
##modelPath = 'D:\\tmp'
# formDist = WorkSheetModelPredict()
#
# dataMap = formDist.getDataMap(config)###需要传给我数据库和表
##relaMap = relaRes[1]###传给我的jison进行转换
#
#
# result = formDist.getDistribution(dataMap, relaMap)
# work_type=formDist.work_type(3,result,configMap1)
# work_vector=formDist.work_vector(result)
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
#
# configMap1 = {}
# configMap1['user'] = 'C##meritdata'
# configMap1['password'] = 'meritdata8991'
# configMap1['host'] = '191.168.6.68'
# configMap1['port'] = '1521'
# configMap1['database'] = 'orcl'

###需要设置
# modelPath = 'D:\\tmp'
#
# dataMap = dataModel.getDataMap(configMap, tableNames)  ###需要传给我数据库和表
# relaMap = relaRes[1]  ###传给我的jison进行转换
#
# formDist = FormDistribution()
# orientTable = formDist.orientTable(relaMap)
# orientCol = formDist.orientCol(relaMap)
# result = formDist.getDistribution(dataMap, tableNames, relaMap, orientTable, orientCol)
# work_type = formDist.work_type(9, result, configMap1)
# work_vector = formDist.work_vector(result)
# print('***************************************')
# for term in result[0]:
#    print(term)
# print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
# print(relaMap)


# relaMap = {('G_SUBS', 'G_SUBS_LINE_RELA'): ('SUBS_ID', 'SUBS_ID', 1),
#            ('G_LINE', 'G_SUBS_LINE_RELA'): ('LINE_ID', 'LINE_ID', 1),
#            ('G_LINE', 'T_TX_ZWYC_XL'): ('PMS_LINE_ID', 'SBID', 1),
#            ('G_SUBS', 'T_TX_ZNYC_DZ'): ('PMS_SUBS_ID', 'SBID',1),
#            ('T_TX_ZWYC_XL', 'T_TX_ZNYC_DZ'):('QSDZ', 'OID', 1)}


# ==============================================================================

# tableNames = ['ARC_S_APP_CERT', 'ARC_S_APP_ELEC_ADDR', 'ARC_S_APP_DATA',
#              'ARC_S_APP_NATURAL_INFO', 'ARC_S_APP_CONTACT', 'ARC_S_APP',
#              'ARC_S_APP_VAT', 'ARC_S_APP_BANK_ACCT', 'ARC_S_APP_ACCT',
#              'ARC_S_APP_PAYMENT_RELA', 'ARC_S_ELEC_DEV_SCHEME']
#
# configMap = {}
# configMap['user'] = 'C##meritdata'
# configMap['password'] = 'meritdata8991'
# configMap['host'] = '191.168.6.68'
# configMap['port'] = '1521'
# configMap['database'] = 'orcl'
#
# orientTable = 'ARC_S_APP'
# orientCol = 'APP_ID'
#
# relaMap = {('ARC_S_APP_CERT', 'ARC_S_APP_NATURAL_INFO'): ('APPN_ID', 'APPN_ID', 1),
#            ('ARC_S_APP_CERT', 'ARC_S_APP'): ('APP_ID', 'APP_ID', 1),
#            ('ARC_S_APP_NATURAL_INFO', 'ARC_S_APP_CONTACT'): ('APPN_ID', 'APPN_ID', 1),
#            ('ARC_S_APP', 'ARC_S_APP_CONTACT'): ('APP_ID', 'APP_ID',1),
#            ('ARC_S_APP_ELEC_ADDR', 'ARC_S_APP'):('APP_ID', 'APP_ID', 1),
#            ('ARC_S_APP_DATA', 'ARC_S_APP'):('APP_ID', 'APP_ID', 1),
#            ('ARC_S_APP_VAT', 'ARC_S_APP'):('APP_ID', 'APP_ID', 1),
#            ('ARC_S_APP_NATURAL_INFO', 'ARC_S_APP_ACCT'):('APPN_ID', 'APPN_ID', 1),
#            ('ARC_S_APP_BANK_ACCT', 'ARC_S_APP_ACCT'):('APP_ACCT_ID', 'APP_ACCT_ID', 1),
#            ('ARC_S_APP_PAYMENT_RELA', 'ARC_S_APP_ACCT'):('APP_ACCT_ID', 'APP_ACCT_ID', 1),
#            ('ARC_S_APP_PAYMENT_RELA', 'ARC_S_APP'):('APP_ID', 'APP_ID', 1),
#            ('ARC_S_ELEC_DEV_SCHEME', 'ARC_S_APP'):('APP_ID', 'APP_ID', 1)}
#
# start = time.clock()
# dist = FormDistribution()
# res = dist.getDistribution(configMap, tableNames, relaMap, orientTable, orientCol)
# end = time.clock()
# print('--------------------------------')
# for term in res[0]:
#    print(term)
# print('================================')
# print(res[1])
# print('================================')
# print('分析计算用时：%s 秒' %(end-start) )
