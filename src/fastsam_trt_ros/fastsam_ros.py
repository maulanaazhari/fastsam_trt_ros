#!/usr/bin/env python

import roslib
roslib.load_manifest("fastsam_trt_ros")

import time
import cv2
import sys
import os
import numpy as np

import rospy
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import message_filters

from fastsam_trt_ros.fastsam import FastSam
from fastsam_trt_ros.tools import *
from fastsam_ros_msgs.msg import DetectionArray, Box, Detection
from fastsam_ros_msgs.utils import *

class FastSamNode:
    def __init__(self, model_path, image_size, compressed=True, conf=0.4, iou=0.7, retina_mask=False, agnostic_nms=False):
        rospy.loginfo("Loading model {}".format(model_path))
        self.cv_image = None
        self.image_size = image_size
        self.conf = conf
        self.iou = iou
        self.retina_mask = retina_mask
        self.agnostic_nms = agnostic_nms
        self.compressed = compressed
        self.img_bridge = CvBridge()
        self.predictor = FastSam(model_path, max_size=image_size)

        # if self.compressed:
        #     self.det_sub = rospy.Subscriber("image_in" +'/compressed', CompressedImage, self.image_callback, queue_size=1)
        #     self.image_pub = rospy.Publisher("image_out" + "/compressed", CompressedImage, queue_size=1)
        # else:
        #     self.image_sub = rospy.Subscriber("image_in", Image, self.image_callback, queue_size=1)
        #     self.image_pub = rospy.Publisher("image_out", Image, queue_size=1)

        self.image_sub = message_filters.Subscriber("image_in/compressed", CompressedImage)
        self.detection_sub = message_filters.Subscriber("detections", Detection2DArray)

        self.image_det_sync = message_filters.TimeSynchronizer([self.image_sub, self.detection_sub], 20)
        self.image_det_sync.registerCallback(self.image_detection_callback)

        self.image_pub = rospy.Publisher("image_out" + "/compressed", CompressedImage, queue_size=1)
        self.detection_pub = rospy.Publisher("fastsam_detections", DetectionArray, queue_size=1)


    def image_detection_callback(self, img_msg:CompressedImage, det_msg:Detection2DArray):
        np_image = np.frombuffer(img_msg.data, dtype=np.uint8)
        self.cv_image = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)
        
        ori_h = self.cv_image.shape[0]
        ori_w = self.cv_image.shape[1]


        # t0 = time.perf_counter()
        self.results = self.predictor.segment(self.cv_image, self.conf, self.iou, self.retina_mask, self.agnostic_nms)
        # print(self.results[0].masks.data[0].shape)

        annotations = []
        for det in det_msg.detections:
            
            c_x = det.bbox.center.x
            c_y = det.bbox.center.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            bbox = [c_x, c_y, w, h]
            # bbox = 
            mask, rbbox, contour, iou_val = box_prompt(self.results[0].masks.data, convert_box_cxcywh_to_xyxy(bbox), ori_h, ori_w)
            
            annotation = {}
            annotation["mask"] = mask.astype(np.int8)
            annotation["bbox"] = bbox
            annotation["rbbox"] = rbbox #format (x0, y0), (w, h), deg
            annotation["score"] = det.results[0].score
            annotation["id"] = det.results[0].id
            annotation["mask_iou"] = iou_val
            annotation["contour"] = contour
            annotations.append(annotation)

        fdet_msg = DetectionArray()
        fdet_msg.header.stamp = det_msg.header.stamp
        fdet_msg.header.frame_id = det_msg.header.frame_id
        fdet_msg.image_height = self.cv_image.shape[0]
        fdet_msg.image_witdh = self.cv_image.shape[1]

        for i, annotation in enumerate(annotations):

            ori_bbox = Box()
            ori_bbox.c_x = annotation["bbox"][0]
            ori_bbox.c_y = annotation["bbox"][1]
            ori_bbox.width = annotation["bbox"][2]
            ori_bbox.height = annotation["bbox"][3]
            ori_bbox.theta = 0.0

            rot_bbox = Box()
            rot_bbox.c_x = annotation["rbbox"][0][0]
            rot_bbox.c_y = annotation["rbbox"][0][1]
            rot_bbox.width = annotation["rbbox"][1][0]
            rot_bbox.height = annotation["rbbox"][1][1]
            rot_bbox.theta = annotation["rbbox"][2]

            det = Detection()
            det.class_id = annotation["id"]
            det.score = annotation["score"]
            det.mask_iou = float(annotation["mask_iou"])
            det.ori_box = ori_bbox
            det.rot_box = rot_bbox
            # det.mask = npMaskToMsg(annotation["mask"])
            poly_x, poly_y = cvContourToPolyXYMsg(annotation["contour"])
            det.polygon_x = poly_x
            det.polygon_y = poly_y

            # print(cvContourFromPolyXYMsg(poly_x, poly_y).shape)
            annotation["contour"] = cvContourFromPolyXYMsg(poly_x, poly_y)

            fdet_msg.detections.append(det)
        
        self.detection_pub.publish(fdet_msg)

        if self.image_pub.get_num_connections() > 0:
            self.disp_img = vis(self.cv_image, annotations)
            # print(self.cv_image.shape   )
            # self.display_img = draw_masks(self.cv_image, self.results, self.image_size)
            disp_msg = CompressedImage()
            disp_msg.header.stamp = img_msg.header.stamp
            disp_msg.format = "jpeg"
            disp_msg.data = np.array(cv2.imencode('.jpg', self.disp_img)[1]).tostring()
            # Publish new image
            self.image_pub.publish(disp_msg)
        # print((time.perf_counter() - t0)*1000, 'ms')

    def image_callback(self, msg):
        
        if self.compressed:
            np_image = np.frombuffer(msg.data, dtype=np.uint8)
            self.cv_image = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)
        else:
            self.cv_image = self.img_bridge.imgmsg_to_cv2(msg, "rgb8")
            if (self.cv_image.shape[2] == 4):
                self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_RGBA2RGB)

        t0 = time.perf_counter()
        self.results = self.predictor.segment(self.cv_image, self.conf, self.iou, self.retina_mask, self.agnostic_nms)
        print((time.perf_counter() - t0)*1000, 'ms')

        if(self.image_pub.get_num_connections() > 0):
            self.display_img = draw_masks(self.cv_image, self.results, self.image_size)
            disp_msg = CompressedImage()
            disp_msg.header.stamp = msg.header.stamp
            disp_msg.format = "jpeg"
            disp_msg.data = np.array(cv2.imencode('.jpg', self.display_img)[1]).tostring()
            # Publish new image
            self.image_pub.publish(disp_msg)

        

def main(args):
    rospy.init_node('fastsam_ros', anonymous=True)
    
    model_path = rospy.get_param('~model_path')
    compressed = rospy.get_param('~compressed', default=False)
    conf = rospy.get_param('~conf', default=0.4)
    iou = rospy.get_param("~iou", default=0.7)
    retina_mask = rospy.get_param("~retina_mask", default=False)
    agnostic_nms = rospy.get_param("~agnostic_nms", default=False)
    image_size = rospy.get_param("~image_size", default=480)

    detector = FastSamNode(model_path, image_size, compressed, conf, iou, retina_mask, agnostic_nms)
    rospy.loginfo("Running fastsam detector!")
    rospy.loginfo("MODEL PATH : {}".format(model_path))
    rospy.loginfo("CONFIDENCE THRESHOLD : {}".format(conf))
    rospy.loginfo("IoU THRESHOLD : {}".format(iou))
    rospy.loginfo("USE RETINA MASK : {}".format(retina_mask))
    rospy.loginfo("USE AGNOSTIC NMS : {}".format(agnostic_nms))
    try:
        rospy.spin()
    except KeyboardInterrupt:
        detector.predictor.destory()
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)