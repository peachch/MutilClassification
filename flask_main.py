from flask import Flask, request, jsonify
import traceback
import re
import yaml
from utils.logger import get_logger
from predict_by_sentence import Predict
# 定义Flask
application = Flask(__name__)
application.config["JSOIN_AS_ASCII"] = False
application.config["JSONIFY_MIMETYPE"] = "application/json;charset=utf-8"

@application.route("/health", methods=["get"])
def health():
    return jsonify({})

@application.route("/pred_sent", methods=["post"])
def main_rec(input):
    try:
        sent_content = request.json["content"]
        assert isinstance(sent_content, str)
    except Exception:
        try:
            sent_content = eval(request.get_data(as_text=True))['content']
            assert isinstance(sent_content, str)
        except Exception as e:
            error_msg = str(e) + "," + re.sub('\n', ' ', traceback.format_exc())
            logger.infor("ERROR: 获取入参【content】异常：{}".format(error_msg))
            error_result = {"resultCode":'1', "resultMessage": error_msg}
            return jsonify(error_result)

    # 到这里说明能取到 test_sent
    try:
        # 利用 predict函数预测输出
        pred_label, pred_cls, pred_result, pre_score = classify_predict.predict(str(sent_content))
        logger.info(str(pred_label))
        logger.info(str(pred_cls))

        result = ({'pred_result': str(pred_result), "pred_cls_index": str(pred_label),
                   "pred_cls":str(pred_cls), 'pred_cls_score':str(pre_score)})
        return jsonify(result)
    except Exception as e:
        error_msg = str(e) + "," + re.sub('\n', ' ', traceback.format_exc())
        logger.info("ERROR:【在线咨询/风控】类别异常：{}".format(error_msg))
        error_result = {"resultCode": '1', "resultMessage":error_msg,
                        "data":{"Content":sent_content, "pred_cls":[]}}
        return jsonify(error_result)

if __name__=="__main__":
    with open("config.yaml", "r") as fp:
        cfg = yaml.safe_load(fp)

    # 载入全局模型，避免每次预测都要载入模型
    log_dir = cfg["train"]["log"]["log_dir"]
    logger = get_logger(
        "train",
        log_dir=log_dir,
        log_filename=cfg["train"]["log"]["log_filename"],
    )
    device = cfg["train"]["device"]
    classify_predict = Predict(cfg, logger, device)
    application.run(host="0.0.0.0", port=8767, threaded=False)