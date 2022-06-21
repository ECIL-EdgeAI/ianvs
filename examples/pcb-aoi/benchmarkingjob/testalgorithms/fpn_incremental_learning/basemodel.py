from __future__ import absolute_import, division, print_function

import os
import tempfile
import time
import logging
import zipfile
import cv2
import pickle

import numpy as np
import tensorflow as tf
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from FPN_TensorFlow.help_utils.help_utils import draw_box_cv
from FPN_TensorFlow.help_utils.tools import *
import tensorflow.contrib.slim as slim
from FPN_TensorFlow.tools import restore_model
from FPN_TensorFlow.libs.label_name_dict.label_dict import LABEl_NAME_MAP, NAME_LABEL_MAP
from FPN_TensorFlow.data.io.convert_to_tfrecord import convert_pascal_to_tfrecord, \
    convert_pascal_to_test_tfrecord
from FPN_TensorFlow.data.io.read_tfrecord import next_batch_for_tasks, convert_labels
from FPN_TensorFlow.data.io import image_preprocess
from FPN_TensorFlow.tools.single_image_eval import main
from FPN_TensorFlow.help_utils.tools import mkdir

LOG = logging.getLogger(__name__)

from libs.configs import cfgs
from libs.box_utils.show_box_in_tensor import (draw_box_with_color,
                                               draw_boxes_with_categories)
from libs.fast_rcnn import build_fast_rcnn
from libs.networks.network_factory import get_flags_byname, get_network_byname
from libs.rpn import build_rpn

FLAGS = get_flags_byname(cfgs.NET_NAME)

# set backend
os.environ['BACKEND_TYPE'] = 'TENSORFLOW'

__all__ = ["BaseModel"]


def preprocess(data_dir, mode):
    # data_dir must include dir "Annotations" and "JPEGImages"
    xml_dir = os.path.join(data_dir, "Annotations")
    image_dir = os.path.join(data_dir, "JPEGImages")
    if mode == "train":
        convert_pascal_to_tfrecord(xml_dir, image_dir)
    else:
        convert_pascal_to_test_tfrecord(xml_dir, image_dir)


@ClassFactory.register(ClassType.GENERAL, "estimator")
class BaseModel:

    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.session = tf.Session(
            config=sess_config, graph=self.graph)

        self.restorer = None
        self.checkpoint_path = self.load(Context.get_parameters("base_model_url"))
        self.temp_dir = tempfile.mkdtemp()
        if not os.path.isdir(self.temp_dir):
            mkdir(self.temp_dir)

        os.environ["MODEL_NAME"] = "model.zip"
        cfgs.LR = kwargs.get("learning_rate", 0.0001)
        cfgs.MOMENTUM = kwargs.get("momentum", 0.9)
        cfgs.MAX_ITERATION = kwargs.get("max_iteration", 5)

    def train(self, train_data, valid_data=None, **kwargs):
        """
        train
        """

        if train_data is None or train_data.x is None or train_data.y is None:
            raise Exception("Train data is None.")

        with tf.Graph().as_default():

            img_name_batch, train_data, gtboxes_and_label_batch, num_objects_batch, data_num = \
                next_batch_for_tasks(
                    (train_data.x, train_data.y),
                    dataset_name=cfgs.DATASET_NAME,
                    batch_size=cfgs.BATCH_SIZE,
                    shortside_len=cfgs.SHORT_SIDE_LEN,
                    is_training=True,
                    save_name="train"
                )

            with tf.name_scope('draw_gtboxes'):
                gtboxes_in_img = draw_box_with_color(train_data, tf.reshape(gtboxes_and_label_batch, [-1, 5])[:, :-1],
                                                     text=tf.shape(gtboxes_and_label_batch)[1])

            # ***********************************************************************************************
            # *                                         share net                                           *
            # ***********************************************************************************************
            _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                              inputs=train_data,
                                              num_classes=None,
                                              is_training=True,
                                              output_stride=None,
                                              global_pool=False,
                                              spatial_squeeze=False)

            # ***********************************************************************************************
            # *                                            rpn                                              *
            # ***********************************************************************************************
            rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                                inputs=train_data,
                                gtboxes_and_label=tf.squeeze(gtboxes_and_label_batch, 0),
                                is_training=True,
                                share_head=cfgs.SHARE_HEAD,
                                share_net=share_net,
                                stride=cfgs.STRIDE,
                                anchor_ratios=cfgs.ANCHOR_RATIOS,
                                anchor_scales=cfgs.ANCHOR_SCALES,
                                scale_factors=cfgs.SCALE_FACTORS,
                                base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                                level=cfgs.LEVEL,
                                top_k_nms=cfgs.RPN_TOP_K_NMS,
                                rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                                max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                                rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                                # iou>=0.7 is positive box, iou< 0.3 is negative
                                rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                                rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                                rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                                remove_outside_anchors=False,  # whether remove anchors outside
                                rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

            rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

            rpn_location_loss, rpn_classification_loss = rpn.rpn_losses()
            rpn_total_loss = rpn_classification_loss + rpn_location_loss

            with tf.name_scope('draw_proposals'):
                # score > 0.5 is object
                rpn_object_boxes_indices = tf.reshape(tf.where(tf.greater(rpn_proposals_scores, 0.5)), [-1])
                rpn_object_boxes = tf.gather(rpn_proposals_boxes, rpn_object_boxes_indices)

                rpn_proposals_objcet_boxes_in_img = draw_box_with_color(train_data, rpn_object_boxes,
                                                                        text=tf.shape(rpn_object_boxes)[0])
                rpn_proposals_boxes_in_img = draw_box_with_color(train_data, rpn_proposals_boxes,
                                                                 text=tf.shape(rpn_proposals_boxes)[0])
            # ***********************************************************************************************
            # *                                         Fast RCNN                                           *
            # ***********************************************************************************************

            fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=train_data,
                                                 feature_pyramid=rpn.feature_pyramid,
                                                 rpn_proposals_boxes=rpn_proposals_boxes,
                                                 rpn_proposals_scores=rpn_proposals_scores,
                                                 img_shape=tf.shape(train_data),
                                                 roi_size=cfgs.ROI_SIZE,
                                                 roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                                 scale_factors=cfgs.SCALE_FACTORS,
                                                 gtboxes_and_label=tf.squeeze(gtboxes_and_label_batch, 0),
                                                 fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                                 fast_rcnn_maximum_boxes_per_img=100,
                                                 fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                                 show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                                 # show detections which score >= 0.6
                                                 num_classes=cfgs.CLASS_NUM,
                                                 fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                                 fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                                 # iou>0.5 is positive, iou<0.5 is negative
                                                 fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                                 use_dropout=False,
                                                 weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                                 is_training=True,
                                                 level=cfgs.LEVEL)

            fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
                fast_rcnn.fast_rcnn_predict()
            fast_rcnn_location_loss, fast_rcnn_classification_loss = fast_rcnn.fast_rcnn_loss()
            fast_rcnn_total_loss = fast_rcnn_location_loss + fast_rcnn_classification_loss

            with tf.name_scope('draw_boxes_with_categories'):
                fast_rcnn_predict_boxes_in_imgs = draw_boxes_with_categories(img_batch=train_data,
                                                                             boxes=fast_rcnn_decode_boxes,
                                                                             labels=detection_category,
                                                                             scores=fast_rcnn_score)

            # train
            added_loss = rpn_total_loss + fast_rcnn_total_loss
            total_loss = tf.losses.get_total_loss()

            global_step = tf.train.get_or_create_global_step()

            lr = tf.train.piecewise_constant(global_step,
                                             boundaries=[np.int64(20000), np.int64(40000)],
                                             values=[cfgs.LR, cfgs.LR / 10, cfgs.LR / 100])
            tf.summary.scalar('lr', lr)
            optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)

            train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)  # rpn_total_loss,
            # train_op = optimizer.minimize(second_classification_loss, global_step)

            # ***********************************************************************************************
            # *                                          Summary                                            *
            # ***********************************************************************************************
            # ground truth and predict
            tf.summary.image('img/gtboxes', gtboxes_in_img)
            tf.summary.image('img/faster_rcnn_predict', fast_rcnn_predict_boxes_in_imgs)
            # rpn loss and image
            tf.summary.scalar('rpn/rpn_location_loss', rpn_location_loss)
            tf.summary.scalar('rpn/rpn_classification_loss', rpn_classification_loss)
            tf.summary.scalar('rpn/rpn_total_loss', rpn_total_loss)

            tf.summary.scalar('fast_rcnn/fast_rcnn_location_loss', fast_rcnn_location_loss)
            tf.summary.scalar('fast_rcnn/fast_rcnn_classification_loss', fast_rcnn_classification_loss)
            tf.summary.scalar('fast_rcnn/fast_rcnn_total_loss', fast_rcnn_total_loss)

            tf.summary.scalar('loss/added_loss', added_loss)
            tf.summary.scalar('loss/total_loss', total_loss)

            tf.summary.image('rpn/rpn_all_boxes', rpn_proposals_boxes_in_img)
            tf.summary.image('rpn/rpn_object_boxes', rpn_proposals_objcet_boxes_in_img)
            # learning_rate
            tf.summary.scalar('learning_rate', lr)

            summary_op = tf.summary.merge_all()
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            # restorer, restore_ckpt = restore_model.get_restorer(test=False, checkpoint_path=kwargs.get("checkpoint_path"))
            restorer, restore_ckpt = restore_model.get_restorer(test=False, checkpoint_path=self.checkpoint_path)
            saver = tf.train.Saver(max_to_keep=3)

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.95
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(init_op)
                if not restorer is None:
                    restorer.restore(sess, restore_ckpt)
                    print('restore model')
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)

                summary_path = os.path.join(self.temp_dir, 'output/{}'.format(cfgs.DATASET_NAME),
                                            FLAGS.summary_path, cfgs.VERSION)

                mkdir(summary_path)
                summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

                for step in range(cfgs.MAX_ITERATION):
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    start = time.time()

                    _global_step, _img_name_batch, _rpn_location_loss, _rpn_classification_loss, \
                    _rpn_total_loss, _fast_rcnn_location_loss, _fast_rcnn_classification_loss, \
                    _fast_rcnn_total_loss, _added_loss, _total_loss, _ = \
                        sess.run([global_step, img_name_batch, rpn_location_loss, rpn_classification_loss,
                                  rpn_total_loss, fast_rcnn_location_loss, fast_rcnn_classification_loss,
                                  fast_rcnn_total_loss, added_loss, total_loss, train_op])

                    end = time.time()

                    if step % 50 == 0:
                        print("""{}: step{} image_name:{}
                             rpn_loc_loss:{:.4f} | rpn_cla_loss:{:.4f} | rpn_total_loss:{:.4f}
                             fast_rcnn_loc_loss:{:.4f} | fast_rcnn_cla_loss:{:.4f} | fast_rcnn_total_loss:{:.4f}
                             added_loss:{:.4f} | total_loss:{:.4f} | pre_cost_time:{:.4f}s"""
                              .format(training_time, _global_step, str(_img_name_batch[0]), _rpn_location_loss,
                                      _rpn_classification_loss, _rpn_total_loss, _fast_rcnn_location_loss,
                                      _fast_rcnn_classification_loss, _fast_rcnn_total_loss, _added_loss, _total_loss,
                                      (end - start)))

                    if step % 500 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, _global_step)
                        summary_writer.flush()

                    if step > 0 and step == cfgs.MAX_ITERATION - 1:
                        self.checkpoint_path = os.path.join(self.temp_dir, '{}_'.format(
                            cfgs.DATASET_NAME) + str(_global_step) + "_" + str(time.time()) + '_model.ckpt')
                        saver.save(sess, self.checkpoint_path)
                        print('Weights have been saved to {}.'.format(self.checkpoint_path))

                print('Training finish.')

                coord.request_stop()
                coord.join(threads)

        return self.checkpoint_path

    def save(self, model_path):
        if not model_path:
            raise Exception("model path is None.")

        model_dir, model_name = os.path.split(self.checkpoint_path)
        models = [model for model in os.listdir(model_dir) if model_name in model]

        if os.path.splitext(model_path)[-1] != ".zip":
            model_path = os.path.join(model_path, "model.zip")

        if not os.path.isdir(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        with zipfile.ZipFile(model_path, "w") as f:
            for model_file in models:
                model_file_path = os.path.join(model_dir, model_file)
                f.write(model_file_path, model_file, compress_type=zipfile.ZIP_DEFLATED)

        return model_path

    def predict(self, data, input_shape=None, **kwargs):

        if data is None:
            raise Exception("Predict data is None")

        inference_output_dir = os.getenv("INFERENCE_OUTPUT_DIR")

        with tf.Graph().as_default():

            img_plac = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

            img_tensor = tf.cast(img_plac, tf.float32) - tf.constant([103.939, 116.779, 123.68])
            img_batch = image_preprocess.short_side_resize_for_inference_data(img_tensor,
                                                                              target_shortside_len=cfgs.SHORT_SIDE_LEN,
                                                                              is_resize=True)

            # ***********************************************************************************************
            # *                                         share net                                           *
            # ***********************************************************************************************
            _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                              inputs=img_batch,
                                              num_classes=None,
                                              is_training=True,
                                              output_stride=None,
                                              global_pool=False,
                                              spatial_squeeze=False)
            # ***********************************************************************************************
            # *                                            RPN                                              *
            # ***********************************************************************************************
            rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                                inputs=img_batch,
                                gtboxes_and_label=None,
                                is_training=False,
                                share_head=cfgs.SHARE_HEAD,
                                share_net=share_net,
                                stride=cfgs.STRIDE,
                                anchor_ratios=cfgs.ANCHOR_RATIOS,
                                anchor_scales=cfgs.ANCHOR_SCALES,
                                scale_factors=cfgs.SCALE_FACTORS,
                                base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                                level=cfgs.LEVEL,
                                top_k_nms=cfgs.RPN_TOP_K_NMS,
                                rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                                max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                                rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                                rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                                rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                                rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                                remove_outside_anchors=False,  # whether remove anchors outside
                                rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

            # rpn predict proposals
            rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

            # ***********************************************************************************************
            # *                                         Fast RCNN                                           *
            # ***********************************************************************************************
            fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=img_batch,
                                                 feature_pyramid=rpn.feature_pyramid,
                                                 rpn_proposals_boxes=rpn_proposals_boxes,
                                                 rpn_proposals_scores=rpn_proposals_scores,
                                                 img_shape=tf.shape(img_batch),
                                                 roi_size=cfgs.ROI_SIZE,
                                                 scale_factors=cfgs.SCALE_FACTORS,
                                                 roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                                 gtboxes_and_label=None,
                                                 fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                                 fast_rcnn_maximum_boxes_per_img=100,
                                                 fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                                 show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                                 # show detections which score >= 0.6
                                                 num_classes=cfgs.CLASS_NUM,
                                                 fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                                 fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                                 fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                                 use_dropout=False,
                                                 weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                                 is_training=False,
                                                 level=cfgs.LEVEL)

            fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
                fast_rcnn.fast_rcnn_predict()

            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            restorer, restore_ckpt = restore_model.get_restorer(checkpoint_path=self.checkpoint_path)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                sess.run(init_op)

                if restorer is not None:
                    restorer.restore(sess, restore_ckpt)
                    print('restore model')

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)

                # boxes_in_all_images = []
                # labels_in_all_images = []
                imgs = [cv2.imread(img) for img in data]
                img_names = [os.path.basename(img_path) for img_path in data]

                predict_dict = {}

                for i, img in enumerate(imgs):
                    start = time.time()

                    _img_batch, _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category = \
                        sess.run([img_batch, fast_rcnn_decode_boxes, fast_rcnn_score, detection_category],
                                 feed_dict={img_plac: img})
                    end = time.time()

                    # predict box dict
                    predict_dict[str(img_names[i])] = []

                    for label in NAME_LABEL_MAP.keys():
                        if label == 'back_ground':
                            continue
                        else:
                            temp_dict = {}
                            temp_dict['name'] = label

                            ind = np.where(_detection_category == NAME_LABEL_MAP[label])[0]
                            temp_boxes = _fast_rcnn_decode_boxes[ind]
                            temp_score = np.reshape(_fast_rcnn_score[ind], [-1, 1])
                            temp_dict['bbox'] = np.array(np.concatenate(
                                [temp_boxes, temp_score], axis=1), np.float64)
                            predict_dict[str(img_names[i])].append(temp_dict)

                    img_np = np.squeeze(_img_batch, axis=0)

                    img_np = draw_box_cv(img_np,
                                         boxes=_fast_rcnn_decode_boxes,
                                         labels=_detection_category,
                                         scores=_fast_rcnn_score)

                    if inference_output_dir:
                        mkdir(inference_output_dir)
                        cv2.imwrite(inference_output_dir + '/{}_fpn.jpg'.format(img_names[i]), img_np)
                        view_bar('{} cost {}s'.format(img_names[i], (end - start)), i + 1, len(imgs))
                        print(f"\nInference results have been saved to directory:{inference_output_dir}.")

                coord.request_stop()
                coord.join(threads)

        return predict_dict

    def load(self, model_url=None):
        if model_url:
            model_dir = os.path.split(model_url)[0]
            with zipfile.ZipFile(model_url, "r") as f:
                f.extractall(path=model_dir)
                ckpt_name = os.path.basename(f.namelist()[0])
                index = ckpt_name.find("ckpt")
                ckpt_name = ckpt_name[:index + 4]
            self.checkpoint_path = os.path.join(model_dir, ckpt_name)

            print(f"load {model_url} to {self.checkpoint_path}")
        else:
            raise Exception(f"model url is None")

        return self.checkpoint_path

    def test(self, valid_data, **kwargs):
        '''
        output the test results and groudtruth
        while this function is not in sedna's incremental learning interfaces
        '''

        checkpoint_path = kwargs.get("checkpoint_path")
        img_name_batch = kwargs.get("img_name_batch")
        gtboxes_and_label_batch = kwargs.get("gtboxes_and_label_batch")
        num_objects_batch = kwargs.get("num_objects_batch")
        graph = kwargs.get("graph")
        data_num = kwargs.get("data_num")

        test.fpn_test(validate_data=valid_data,
                      checkpoint_path=checkpoint_path,
                      graph=graph,
                      img_name_batch=img_name_batch,
                      gtboxes_and_label_batch=gtboxes_and_label_batch,
                      num_objects_batch=num_objects_batch,
                      data_num=data_num)

    def evaluate(self, data, model_path, **kwargs):
        if data is None or data.x is None or data.y is None:
            raise Exception("Prediction data is None")

        self.load(model_path)
        predict_dict = self.predict(data.x)

        metric = kwargs.get("metric")
        if callable(metric):
            return {"f1_score": metric(data.y, predict_dict)}
        return {"f1_score": f1_score(data.y, predict_dict)}

    def fpn_eval_dict_convert(self, data, checkpoint_path=None):

        with tf.Graph().as_default():

            img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch, data_num = \
                next_batch_for_tasks(
                    (data.x, data.y),
                    dataset_name=cfgs.DATASET_NAME,
                    batch_size=cfgs.BATCH_SIZE,
                    shortside_len=cfgs.SHORT_SIDE_LEN,
                    save_name="test",
                    is_training=False
                )

            # ***********************************************************************************************
            # *                                         share net                                           *
            # ***********************************************************************************************
            _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                              inputs=img_batch,
                                              num_classes=None,
                                              is_training=True,
                                              output_stride=None,
                                              global_pool=False,
                                              spatial_squeeze=False)

            # ***********************************************************************************************
            # *                                            RPN                                              *
            # ***********************************************************************************************
            rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                                inputs=img_batch,
                                gtboxes_and_label=None,
                                is_training=False,
                                share_head=True,
                                share_net=share_net,
                                stride=cfgs.STRIDE,
                                anchor_ratios=cfgs.ANCHOR_RATIOS,
                                anchor_scales=cfgs.ANCHOR_SCALES,
                                scale_factors=cfgs.SCALE_FACTORS,
                                base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                                level=cfgs.LEVEL,
                                top_k_nms=cfgs.RPN_TOP_K_NMS,
                                rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                                max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                                rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                                rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                                rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                                rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                                remove_outside_anchors=False,  # whether remove anchors outside
                                rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

            # rpn predict proposals
            rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

            # ***********************************************************************************************
            # *                                         Fast RCNN                                           *
            # ***********************************************************************************************
            fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=img_batch,
                                                 feature_pyramid=rpn.feature_pyramid,
                                                 rpn_proposals_boxes=rpn_proposals_boxes,
                                                 rpn_proposals_scores=rpn_proposals_scores,
                                                 img_shape=tf.shape(img_batch),
                                                 roi_size=cfgs.ROI_SIZE,
                                                 scale_factors=cfgs.SCALE_FACTORS,
                                                 roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                                 gtboxes_and_label=None,
                                                 fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                                 fast_rcnn_maximum_boxes_per_img=100,
                                                 fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                                 show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                                 # show detections which score >= 0.6
                                                 num_classes=cfgs.CLASS_NUM,
                                                 fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                                 fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                                 fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                                 use_dropout=False,
                                                 weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                                 is_training=False,
                                                 level=cfgs.LEVEL)

            fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
                fast_rcnn.fast_rcnn_predict()

            # train
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            if checkpoint_path:
                restorer, restore_ckpt = restore_model.get_restorer(checkpoint_path=checkpoint_path)
            else:
                checkpoint_path = tf.train.latest_checkpoint(
                    os.path.join('output/{}'.format(cfgs.DATASET_NAME), FLAGS.trained_checkpoint, cfgs.VERSION))
                print(f'When weight path is not specified, use the latest weight from {checkpoint_path}')

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(init_op)
                if not restorer is None:
                    restorer.restore(sess, restore_ckpt)
                    print('restore model')

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)

                gtbox_dict = {}
                predict_dict = {}

                for i in range(data_num):
                    start = time.time()

                    _img_name_batch, _img_batch, _gtboxes_and_label_batch, _fast_rcnn_decode_boxes, \
                    _fast_rcnn_score, _detection_category \
                        = sess.run([img_name_batch, img_batch, gtboxes_and_label_batch, fast_rcnn_decode_boxes,
                                    fast_rcnn_score, detection_category])
                    end = time.time()

                    # gtboxes convert dict
                    gtbox_dict[str(_img_name_batch[0])] = []
                    predict_dict[str(_img_name_batch[0])] = []

                    for j, box in enumerate(_gtboxes_and_label_batch[0]):
                        bbox_dict = {}
                        bbox_dict['bbox'] = np.array(_gtboxes_and_label_batch[0][j, :-1], np.float64)
                        bbox_dict['name'] = LABEl_NAME_MAP[int(_gtboxes_and_label_batch[0][j, -1])]
                        gtbox_dict[str(_img_name_batch[0])].append(bbox_dict)

                    for label in NAME_LABEL_MAP.keys():
                        if label == 'back_ground':
                            continue
                        else:
                            temp_dict = {}
                            temp_dict['name'] = label

                            ind = np.where(_detection_category == NAME_LABEL_MAP[label])[0]
                            temp_boxes = _fast_rcnn_decode_boxes[ind]
                            temp_score = np.reshape(_fast_rcnn_score[ind], [-1, 1])
                            temp_dict['bbox'] = np.array(np.concatenate(
                                [temp_boxes, temp_score], axis=1), np.float64)
                            predict_dict[str(_img_name_batch[0])].append(temp_dict)

                    view_bar('{} image cost {}s'.format(
                        str(_img_name_batch[0]), (end - start)), i + 1, data_num)

                fw1 = open('gtboxes_dict.pkl', 'wb')
                fw2 = open('predict_dict.pkl', 'wb')
                pickle.dump(gtbox_dict, fw1)
                pickle.dump(predict_dict, fw2)
                fw1.close()
                fw2.close()
                coord.request_stop()
                coord.join(threads)

        return predict_dict, gtbox_dict

    def task_division(self, samples, threshold):

        return main(samples, threshold)


def f1_score(y_true, y_pred):
    predict_dict = {}

    for k, v in y_pred.items():
        k = f"b'{k}'"
        if not predict_dict.get(k):
            predict_dict[k] = v

    gtboxes_dict = convert_labels(y_true)

    R, P, AP, F, num = [], [], [], [], []

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue

        rboxes, gboxes = get_single_label_dict(predict_dict, gtboxes_dict, label)
        # print('label',label)
        rec, prec, ap, box_num = single_label_eval(rboxes, gboxes, 0.3, False)
        # print("rec",rec)
        # print("prec", prec)
        recall = 0 if rec.shape[0] == 0 else rec[-1]
        precision = 0 if prec.shape[0] == 0 else prec[-1]
        F_measure = 0 if not (recall + precision) else (2 * precision * recall / (recall + precision))
        print('\n{}\tR:{}\tP:{}\tap:{}\tF:{}'.format(label, recall, precision, ap, F_measure))
        R.append(recall)
        P.append(precision)
        AP.append(ap)
        F.append(F_measure)
        num.append(box_num)
    print("num:", num)
    R = np.array(R)
    P = np.array(P)
    AP = np.array(AP)
    F = np.array(F)
    num = np.array(num)
    weights = num / np.sum(num)
    Recall = np.sum(R) / 2
    Precision = np.sum(P) / 2
    mAP = np.sum(AP) / 2
    F_measure = np.sum(F) / 2
    print('\n{}\tR:{}\tP:{}\tmAP:{}\tF:{}'.format('Final', Recall, Precision, mAP, F_measure))

    # return F_measure
    return Recall

def get_single_label_dict(predict_dict, gtboxes_dict, label):
    rboxes = {}
    gboxes = {}
    rbox_images = predict_dict.keys()
    rbox_images = list(rbox_images)

    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        for pre_box in predict_dict[rbox_image]:
            if pre_box['name'] == label and len(pre_box['bbox']) != 0:
                rboxes[rbox_image] = [pre_box]

                gboxes[rbox_image] = []

                for gt_box in gtboxes_dict[rbox_image]:
                    if gt_box['name'] == label:
                        gboxes[rbox_image].append(gt_box)
    return rboxes, gboxes


def single_label_eval(rboxes, gboxes, iou_th, use_07_metric):
    rbox_images = list(rboxes.keys())
    fp = np.zeros(len(rbox_images))
    tp = np.zeros(len(rbox_images))
    box_num = 0

    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        if len(rboxes[rbox_image][0]['bbox']) > 0:

            rbox_lists = np.array(rboxes[rbox_image][0]['bbox'])
            if len(gboxes[rbox_image]) > 0:
                gbox_list = np.array([obj['bbox'] for obj in gboxes[rbox_image]])
                box_num = box_num + len(gbox_list)
                gbox_list = np.concatenate((gbox_list, np.zeros((np.shape(gbox_list)[0], 1))), axis=1)
                confidence = rbox_lists[:, 4]
                box_index = np.argsort(-confidence)

                rbox_lists = rbox_lists[box_index, :]
                for rbox_list in rbox_lists:

                    ixmin = np.maximum(gbox_list[:, 0], rbox_list[0])
                    iymin = np.maximum(gbox_list[:, 1], rbox_list[1])
                    ixmax = np.minimum(gbox_list[:, 2], rbox_list[2])
                    iymax = np.minimum(gbox_list[:, 3], rbox_list[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((rbox_list[2] - rbox_list[0] + 1.) * (rbox_list[3] - rbox_list[1] + 1.) +
                           (gbox_list[:, 2] - gbox_list[:, 0] + 1.) *
                           (gbox_list[:, 3] - gbox_list[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                    if ovmax > iou_th:
                        if gbox_list[jmax, -1] == 0:
                            tp[i] += 1
                            gbox_list[jmax, -1] = 1
                        else:
                            fp[i] += 1
                    else:
                        fp[i] += 1

            else:
                fp[i] += len(rboxes[rbox_image][0]['bbox'])
        else:
            continue
    rec = np.zeros(len(rbox_images))
    prec = np.zeros(len(rbox_images))
    if box_num == 0:
        for i in range(len(fp)):
            if fp[i] != 0:
                prec[i] = 0
            else:
                prec[i] = 1

    else:

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        rec = tp / box_num

    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, box_num


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
