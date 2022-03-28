#!/usr/bin/env python
#--coding:utf-8--
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import numpy as np



kernel = np.ones((7,7),np.uint8)
rect=[0 for x in range(0, 40)]


kernel2 = np.ones((11,11),np.uint8)
lower=np.array([0,50,50])
upper=np.array([5,250,255])

def nothing(x):  # 滑动条的回调函数
    pass


def detect(c):
        # 初始化形状名称，使用轮廓近似法
        shape = "unidentified"
        # 计算周长
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # 轮廓是由一系列顶点组成的；如果是三角形，将拥有3个向量
        if len(approx) == 3:
            shape = "triangle"
        # 如果有4个顶点，那么是矩形或者正方形
        elif len(approx) == 4:
            # 计算轮廓的边界框 并且计算宽高比
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # 正方形的宽高比~~1 ，否则是矩形
            shape = "square" if ar >= 0.85 and ar <= 1.15 else "rectangle"
        # 如果是五边形（pentagon），将有5个顶点
        elif len(approx) == 5:
            shape = "pentagon"
        # 否则，根据上边的膨胀腐蚀，我们假设它为圆形
        elif len(approx) > 7 :
            shape = "circle"
        # 返回形状的名称
        return shape

def find_line(img):
    #lower_yellow=np.array([26,43,56])           
    lower_yellow=np.array([15,43,156])                      
    upper_yellow=np.array([34,255,255])
    img=cv2.GaussianBlur(img,(11,11),0)
    #转HSV
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    shape2=['empty'for x in range(0, 20)]
    last_aero_line = 0
    mask=cv2.inRange(hsv,lower_yellow,upper_yellow)   
    #去除噪点
    #腐蚀
    mask=cv2.erode(mask,kernel,iterations=1)
    #膨胀
    mask=cv2.dilate(mask,kernel2,iterations=2)
    mask=cv2.dilate(mask,kernel,iterations=1)
    mask[0:3,0:860] = 0
    mask[477:480,0:860] = 0
    mask[0:480,0:3] = 0
    mask[0:480,857:860] = 0
    #RETR_EXTERNAL只检测最外围的轮廓  CHAIN_APPROX_SIMPLE只存取四个角点
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2] 
    len_cnts_line=len(cnts)
 
    if len(cnts)>0:
     for i in range(len(cnts)):
        shape2[i] = detect(cnts[i])
        rect[i]=cv2.minAreaRect(cnts[i])
        all_yellow_contourArea [i] = cv2.contourArea(cnts[i])
        if all_yellow_contourArea [i]>last_aero_line:
            max_yellow_contourArea_index=i
            last_aero_line=all_yellow_contourArea [i]
     #max_yellow_contourArea_index,max_yellow_contourArea_aera = max(enumerate(all_yellow_contourArea),key=operator.itemgetter(1))
     return len_cnts_line,max_yellow_contourArea_index,rect[max_yellow_contourArea_index],cnts[max_yellow_contourArea_index],shape2[max_yellow_contourArea_index]
    else:
	    return 0,0,[[0,0],[0,0],0],0,'empty'

     
class img_transform:
    Trackbar_flag=0
    def __init__(self):
        self.cam_ros = 0
        self.bridge=CvBridge()
        if self.cam_ros:
            camera = cv2.VideoCapture(0)
            while True:
                # 读取当前帧
                ret, frame = camera.read()
                if ret:
                # 转为灰度图像
                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.callback(frame)==0:
                        break
        else:
            rospy.init_node('ur_image_node_track', anonymous=True)
            #self.image_sub = rospy.Subscriber('/rgb_camera/image_raw',Image,self.callback)
            self.image_sub = rospy.Subscriber('/camera/color/image_rect_color',Image,self.callback)
            
        # camera = cv2.VideoCapture(0)
        # while True:
        #     # 读取当前帧
        #     ret, frame = camera.read()
        #     if ret:
        #     # 转为灰度图像
        #         #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #         if self.callback(frame)==0:
        #             break
               
                
        cv2.destroyAllWindows()
    def callback(self,data):

        kernel = np.ones((7,7),np.uint8)
        kernel2 = np.ones((21,21),np.uint8)
        global num
        global focalLength, control,i,control_last
        if self.cam_ros:
            cv_image=data
        else:    
            cv_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        #初始化三个环的PID参数
        # print("get")
        try:
            
            #cv_image=data
            find_ball_code = 0 
            find_line_code = 1
            if find_line_code == 1:
                WindowName_line = 'yellow_line'  # 窗口名
                cv2.namedWindow(WindowName_line, cv2.WINDOW_AUTOSIZE)  # 建立空窗口
                if self.Trackbar_flag==0:
                    cv2.createTrackbar('h_low', WindowName_line, 15, 255, nothing)  # 创建滑动条
                    cv2.createTrackbar('h_high', WindowName_line, 34, 255, nothing)  # 创建滑动条
                    cv2.createTrackbar('s_low', WindowName_line, 50, 255, nothing)  # 创建滑动条
                    cv2.createTrackbar('v_low', WindowName_line, 50, 255, nothing)  # 创建滑动条
                    cv2.createTrackbar('kernel1', WindowName_line, 0, 50, nothing)  # 创建滑动条
                    cv2.createTrackbar('kernel2', WindowName_line, 0, 50, nothing)  # 创建滑动条
                    cv2.createTrackbar('iterations1', WindowName_line, 0, 50, nothing)  # 创建滑动条
                    cv2.createTrackbar('iterations2', WindowName_line, 0, 50, nothing)  # 创建滑动条
                    self.Trackbar_flag=1
                h_low_line = cv2.getTrackbarPos('h_low', WindowName_line)  # 获取滑动条值
                h_high_line = cv2.getTrackbarPos('h_high', WindowName_line)  # 获取滑动条值
                s_low_line = cv2.getTrackbarPos('s_low', WindowName_line)  # 获取滑动条值
                v_low_line = cv2.getTrackbarPos('v_low', WindowName_line)  # 获取滑动条值
                kernel1_line = cv2.getTrackbarPos('kernel1', WindowName_line)  # 获取滑动条值
                kernel2_line = cv2.getTrackbarPos('kernel2', WindowName_line)  # 获取滑动条值
                iterations1_line = cv2.getTrackbarPos('iterations1', WindowName_line)  # 获取滑动条值
                iterations2_line = cv2.getTrackbarPos('iterations2', WindowName_line)  # 获取滑动条值
                kernel_line = np.ones((kernel1_line,kernel1_line),np.uint8)
                kernel_2_line = np.ones((kernel2_line,kernel2_line),np.uint8)
                lower_yellow=np.array([h_low_line,s_low_line,v_low_line])                      
                upper_yellow=np.array([h_high_line,255,255])
                img=cv2.GaussianBlur(cv_image,(11,11),0)
                hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
      
                mask=cv2.inRange(hsv,lower_yellow,upper_yellow)   

                mask=cv2.erode(mask,kernel_line,iterations1_line)
                mask=cv2.dilate(mask,kernel_2_line,iterations2_line)

                # mask[0:3,0:860] = 0
                # mask[477:480,0:860] = 0
                # mask[0:480,0:3] = 0
                # mask[0:480,857:860] = 0
 
                cv2.imshow(WindowName_line,mask)
                if cv2.waitKey(5) & 0xFF == 27:

                    return 0

     
        
        except CvBridgeError as e :
            print(e)
   
       
if __name__ == '__main__':




    ic=img_transform()
    

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("over!")
    cv2.destroyAllWindows()
    
