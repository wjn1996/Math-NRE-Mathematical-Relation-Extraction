#应用：对文本内标注的实体两两进行关系分类
import tensorflow as tf
import numpy as np
import os
import subprocess

import data_helpers
import utils
from configure import FLAGS
import pre_rec_f1 as prf
import sys

# x_text = ['扇 形 面 积 S = n π R ² / 3 6 0 = L R / 2 （ L 为 扇 形 的 弧 长','y 轴 ， y 轴 上 的 实 数 表 示 纵 坐 标 ） ， 在 函 数 中 表 示 因 变 量']
# x_text = []
def evaluate(x_text):
    # with tf.device('/cpu:0'):
    #     x_text, y = data_helpers.load_data_and_labels(FLAGS.test_path)
    # x_text = eval(FLAGS.app_test)
    print(x_text)
    # Map data into vocabulary  
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    x = np.array(list(text_vocab_processor.transform(x_text)))

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
            rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            preds = []
            for x_batch in batches:
                pred = sess.run(predictions, {input_text: x_batch,
                                              emb_dropout_keep_prob: 1.0,
                                              rnn_dropout_keep_prob: 1.0,
                                              dropout_keep_prob: 1.0})
                preds.append(pred)
            preds = np.concatenate(preds)
            print('preds=',preds)
            # truths = np.argmax(y, axis=1)
            # #保存预测和实际的类标
            # prediction_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictions.txt")
            # truth_path = os.path.join(FLAGS.checkpoint_dir, "..", "ground_truths.txt")
            # prediction_file = open(prediction_path, 'w')
            # truth_file = open(truth_path, 'w')
            # for i in range(len(preds)):
            #     prediction_file.write("{} {}\n".format(i, utils.label2class[preds[i]]))
            #     truth_file.write("{} {}\n".format(i, utils.label2class[truths[i]]))
            # prediction_file.close()
            # truth_file.close()
            # #perl脚本计算精度、召回率和F1
            # perl_path = os.path.join(os.path.curdir,
            #                          "SemEval2010_task8_all_data",
            #                          "SemEval2010_task8_scorer-v1.2",
            #                          "semeval2010_task8_scorer-v1.2.pl")
            # process = subprocess.Popen(["perl", perl_path, prediction_path, truth_path], stdout=subprocess.PIPE)
            # for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
            #     print(line)
            # # python计算精度、召回率和F1 add by wjn
            # print('prediction_path=',prediction_path)
            # prf.main(FLAGS.model_dir + 'predictions.txt',FLAGS.model_dir + 'ground_truths.txt')

def main(_):
    # x_text = input("输入句子")
    # evaluate()
    pass


if __name__ == "__main__":
    # txt = sys.argv[1]
    # x_text.append(txt)
    # tf.app.run()
    x_text = ['三 角 形 的 面 积']
    evaluate(x_text)