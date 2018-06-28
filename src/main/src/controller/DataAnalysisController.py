from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
from analysis.FeatureAndTrain import TableFeatureAndTrain
from analysis.DataModelPredict import DataModel
from analysis.FieldFeatureAndTrain import FieldMatchTrain
from analysis.FieldMatchPredict import FieldMatchApply
from analysis.DbConfig import DbConfig
from analysis.WorkSheetModelPredict import WorkSheetModelPredict

import cx_Oracle, uuid, datetime, threading

app = Flask(__name__)
api = Api(app)
uuidUtil = uuid

parser = reqparse.RequestParser()
parser.add_argument('paramJson', type=dict, required=True)


class TableModelTrainController(Resource):
    def post(self):
        try:
            print("Start run feature train.")
            paramJson = parser.parse_args()
            jsonData = paramJson['paramJson']
            dbConfig = jsonData['dbConfig']
            trainFilePath = jsonData['trainFilePath']
            modelPath = jsonData['modelPath']
            tableFeatureAndTrain = TableFeatureAndTrain()
            tableFeatureAndTrain.runFeatureTrain(trainFilePath, modelPath, dbConfig)
            print("End run feature train.")
        except BaseException as err:
            return {'status': '-1', 'msg': err.__str__()}
        else:
            return {'status': '0', 'msg': 'success'}


class TableModelApplyController(Resource):
    # def updateResult(self, id, status, result):
    #     config = DbConfig()
    #     url = config.host + ':' + config.port + '/' + config.db
    #     try:
    #         conn = cx_Oracle.connect(config.user, config.pwd, url)
    #         cursor = conn.cursor()
    #         param = [status, datetime.datetime.now(), result, id]
    #         sql = "UPDATE pf_data_model_config SET status = :1, update_time = :2,analyse_result = :3 WHERE id = :4"
    #         cursor.execute(sql, param)
    #         conn.commit()
    #     except BaseException as err:
    #         print(err)
    #     finally:
    #         cursor.close()
    #         conn.close()

    # def runProcess(self, data):
    #     id = ''
    #     status = 3  # success
    #     try:
    #         print(data)
    #         jsonData = data['paramJson']
    #         id = jsonData['id']
    #         modelPath = jsonData['modelPath']
    #         tables = jsonData['tables']
    #         params = jsonData['params']
    #         dataModel = DataModel()
    #         result = dataModel.runProcess(params, tables, modelPath)
    #         print(result)
    #     except BaseException as err:
    #         print(err)
    #         result = err.__str__()
    #         status = -1  # failed
    #         self.updateResult(id, status, result)
    #     else:
    #         self.updateResult(id, status, result)
    #     return status, result

    def post(self):
        print("Start apply table model.")
        data = parser.parse_args()
        # threading
        # t = threading.Thread(target=self.runProcess, args=(data,))
        # t.setDaemon(True)
        # t.start()

        try:
            print("parms is : {}".format(data))
            jsonData = data['paramJson']
            modelPath = jsonData['modelPath']
            tables = jsonData['tables']
            params = jsonData['params']
            dataModel = DataModel()
            result = dataModel.runProcess(params, tables, modelPath)
        except BaseException as err:
            print(err)
            msg = err.__str__()
            return {'status': -1, 'msg': msg, 'data': result}
        else:
            return {'status': 0, 'msg': 'success', 'data': result}
        finally:
            print("End apply table model.")


class FlowTableModelApplyController(Resource):
    def updateResult(self, id, status, result):
        config = DbConfig()
        url = config.host + ':' + config.port + '/' + config.db
        try:
            conn = cx_Oracle.connect(config.user, config.pwd, url)
            cursor = conn.cursor()
            param = [status, datetime.datetime.now(), result, id]
            sql = "UPDATE pf_data_model_flow SET analyse_status = :1, update_time = :2,analyse_result = :3 WHERE id = :4"
            cursor.execute(sql, param)
            conn.commit()
        except BaseException as err:
            print(err)
        finally:
            cursor.close()
            conn.close()

    def runProcess(self, data):
        id = ''
        status = 3  # 分析完成
        try:
            print(data)
            jsonData = data['paramJson']
            id = jsonData['id']
            modelPath = jsonData['modelPath']
            tables = jsonData['tables']
            params = jsonData['params']
            print("Start apply flow table model. id : %s" % id)
            dataModel = DataModel()
            result = dataModel.runProcess(params, tables, modelPath)
            print(result)
            print("End apply flow table model.")
        except BaseException as err:
            print(err)
            result = err.__str__()
            status = -1
            self.updateResult(id, status, result)
        else:
            self.updateResult(id, status, result)

    def post(self):
        data = parser.parse_args()
        # threading
        t = threading.Thread(target=self.runProcess, args=(data,))
        t.setDaemon(True)
        t.start()
        return {'status': '0', 'msg': 'success'}


class FieldMatchTrainController(Resource):
    def post(self):
        try:
            print("Start run field feature train.")
            paramJson = parser.parse_args()
            jsonData = paramJson['paramJson']
            dbConfig = jsonData['dbConfig']
            trainFilePath = jsonData['trainFilePath']
            modelPath = jsonData['modelPath']
            fieldMatchTrain = FieldMatchTrain()
            fieldMatchTrain.trainAndSave(trainFilePath, modelPath, dbConfig)
            print("End run field feature train.")
        except BaseException as err:
            print(err)
            return {'status': '-1', 'msg': err.__str__()}
        else:
            return {'status': '0', 'msg': 'success'}


class FieldMatchApplyController(Resource):
    def post(self):
        result = ''
        try:
            print("Start apply filed model.")
            paramJson = parser.parse_args()
            jsonData = paramJson['paramJson']
            dbConfigs = jsonData['dbConfigs']
            for dbc in dbConfigs:
                dbc['tables'] = dbc['tables'].split(',')
            # dbConfig ['tables'] = ['table1','table2']
            #
            #
            # []
            # table1 = jsonData['table1']
            # table2 = jsonData['table2']
            # table3 = jsonData['table3']
            modelPath = jsonData['modelPath']
            fieldMatchApply = FieldMatchApply()
            result = fieldMatchApply.fieldPred(dbConfigs, modelPath)
            print("End apply field model.")
            # make result json
            # resultJson = {}
            # resultJson['status'] = '0'
            # resultJson['msg'] = 'success'
            # resultJson['data'] = result
            # print(resultJson)
            # r = demjson.encode(resultJson)
            print(result)
        except Exception as err:
            print(err)
            return {'status': '-1', 'msg': err.__str__()}
        else:
            return {'data': result, 'status': '0', 'msg': 'success'}


class WorkSheetApplyController(Resource):
    def runProcess(self, data):
        print("Start apply work sheet model.")
        execStatus = 3
        status = -1
        try:
            jsonData = data['paramJson']
            print(jsonData)
            taskId = jsonData['taskId']
            flowId = jsonData['flowId']
            print("taskId : %s flowId : %s" % (taskId, flowId))
            tables = jsonData['tables']
            params = jsonData['params']

            config = DbConfig()

            configMap1 = {}
            configMap1['user'] = config.user
            configMap1['password'] = config.pwd
            configMap1['host'] = config.host
            configMap1['port'] = config.port
            configMap1['database'] = config.db
            relTabs = jsonData['rel']

            relaMap = {}
            for i in range(len(relTabs)):
                relTab = relTabs[i]
                tabs = relTab['tabs'].split(',')
                cols = relTab['cols'].split(',')
                relaMap[(tabs[0], tabs[1])] = (cols[0], cols[1], 1)

            print(relaMap)

            formDist = WorkSheetModelPredict()
            result = formDist.runProcess(flowId, params, relaMap, configMap1)
        except BaseException as err:
            print(err)
            result = err.__str__()
            self.updateResult(taskId, flowId, status, result)
        else:
            status = 0
            self.updateResult(taskId, flowId, status, result)
        print("End apply work sheet model.")

    def post(self):
        data = parser.parse_args()
        # threading
        t = threading.Thread(target=self.runProcess, args=(data,))
        t.setDaemon(True)
        t.start()
        return {'status': '0', 'msg': 'success'}

    def updateResult(self, taskId, flowId, status, result):
        # update status
        config = DbConfig()
        url = config.host + ':' + config.port + '/' + config.db
        try:
            conn = cx_Oracle.connect(config.user, config.pwd, url)
            cursor = conn.cursor()
            param = [status, datetime.datetime.now(), taskId]
            sql = "UPDATE pf_worksheet_analyse_task SET analyse_status = :1, update_time = :2 WHERE id = :3"
            print("update sql : {}".format(sql))
            cursor.execute(sql, param)
            uuid = uuidUtil.uuid1().__str__()
            param = [uuid, taskId, flowId, result, datetime.datetime.now(), status]
            sql = "INSERT INTO pf_worksheet_analyse_result(id,task_id,flow_id,analyse_result,create_time,analyse_status) VALUES(:1,:2,:3,:4,:5,:6)"
            print("insert sql : {}".format(sql))
            cursor.execute(sql, param)
            conn.commit()
        except BaseException as err:
            print(err)
        finally:
            cursor.close()
            conn.close()


api.add_resource(TableModelTrainController, '/tableModelTrain')
api.add_resource(TableModelApplyController, '/tableModelApply')
api.add_resource(FlowTableModelApplyController, '/flowTableModelApply')
api.add_resource(FieldMatchTrainController, '/fieldModelTrain')
api.add_resource(FieldMatchApplyController, '/fieldModelApply')
api.add_resource(WorkSheetApplyController, '/workSheetTableModelApply')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444)
