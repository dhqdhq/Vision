# -*- coding: utf-8 -*-
# 测试使用opencv中的函数solvepnp
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import cv2.aruco as aruco

def nothing(x):
    pass
 
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.createTrackbar('threshold_low', 'image', 0, 500, nothing)
cv2.createTrackbar('threshold_high', 'image', 50, 255, nothing)
cv2.createTrackbar('canny_low', 'image', 0, 255, nothing)
cv2.createTrackbar('canny_high', 'image', 0, 255, nothing)
cv2.createTrackbar('kernel', 'image', 3, 11, nothing)
cv2.createTrackbar('k', 'image', 1, 500, nothing)

def angle(x1, y1, x2, y2):
    if x1 == x2:
        return 90
    if y1 == y2:
        return 180
    k = -(y2 - y1) / (x2 - x1)
    # 求反正切，再将得到的弧度转换为度
    result = np.arctan(k) * 57.29577
    # 234象限
    if x1 > x2 and y1 > y2:
        result += 180
    elif x1 > x2 and y1 < y2:
        result += 180
    elif x1 < x2 and y1 < y2:
        result += 360
    # print("直线倾斜角度为：" + str(result) + "度")
    return result

def points_get(img):

    threshold_low = cv2.getTrackbarPos('threshold_low', 'image')
    threshold_high = cv2.getTrackbarPos('threshold_high', 'image')
    canny_low = cv2.getTrackbarPos('canny_low', 'image')
    canny_high = cv2.getTrackbarPos('canny_high', 'image')
    kernel = cv2.getTrackbarPos('kernel', 'image')
    k = cv2.getTrackbarPos('k', 'image')
    k = k/100.
    kernel +=1
    b = cv2.getTrackbarPos('B', 'image')
    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  #转灰度图
    # print(imgGray.shape)
    
   
    # imgBlur = cv2.GaussianBlur(imgGray,(11,11),1)  #高斯模糊
    # cv2.imshow("imageimgGray",imgGray)
    dst,imgthr = cv2.threshold(imgGray, 254, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_1 = cv2.morphologyEx(imgthr, cv2.MORPH_OPEN, kernel)
    cv2.imshow("binary_1",binary_1)
    img_canny = cv2.Canny(binary_1,canny_low,canny_high)  #Canny算子边缘检测
    cv2.imshow("canny",img_canny)
    contours,hierarchy = cv2.findContours(img_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    imgContour = img
    areas = []
    points = []
    for obj in contours:
        # shape = detect(obj)
        areas.append(cv2.contourArea(obj))  #计算轮廓内区域的面积
    areas.sort(reverse = True)
    if(len(areas)>3):
        for obj in contours:
            # shape = detect(obj)
            area = cv2.contourArea(obj)  #计算轮廓内区域的面积
            if(area>=areas[3]):
                perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
                approx = cv2.approxPolyDP(obj,0.04*perimeter,True)  #获取轮廓角点坐标
                CornerNum = len(approx)   #轮廓角点的数量
                objType ="none"
                # #轮廓对象分类
                
            
                x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度
                # cv2.rectangle(imgContour,(x,y),(x+w,y+h),(100,0,255),5)  #绘制边界框
                
                # cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
                rect = cv2.minAreaRect(obj)
                if area == areas[0]:
                    points.insert(0,rect[0])
                else:
                    points.append(rect[0])
                # 得到最小矩形的坐标
                box = cv2.boxPoints(rect)
                # 标准化坐标到整数
                box = np.int0(box)
                # 画出边界
                
                objType ="rectriangle"
                cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 3)
                if area == areas[0]:
                  cv2.putText(imgContour,str(CornerNum),(x+(w//2),y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),1)  #绘制文字

                # elif CornerNum>4: objType= "Circle"
                # else:objType="N"
    cv2.imshow("image",imgthr)
    # print(points)
    return points




# point2_2D = []
# for i in range(4):
#     p = points1[i]
#     point2_2D.append(points2[i])
# point2_2D = np.array(point2_2D).astype(np.float64)


# cap.set(3,1280)
# cap.set(4,720)
# 738.272035 0.000000 597.802407
# 0.000000 738.646167 357.959193
# 0.000000 0.000000 1.000000

# distortion
# 0.104831 -0.139961 -0.000401 0.000436 0.000000

# cap.set(3,640)
# cap.set(4,480)
# 447.392202 0.000000 296.278715
# 0.000000 447.493062 240.605285
# 0.000000 0.000000 1.000000

# distortion
# 0.112690 -0.150897 0.000098 0.000475 0.000000

#定义回调函数，参数x为函数cv2.createTrackbar()传递的滑块位置对应的值


def main():
    cap = cv2.VideoCapture(1)

    dist=np.array(([[0.112690 ,-0.150897 ,0.000098, 0.000475, 0.000000]]))

    K=np.array([[447.392202 ,0.000000, 296.278715],
                            [ 0.000000, 447.493062 ,240.605285],
                            [  0.,           0.,           1.        ]], dtype=np.float64)
    objPoints = np.array([[0, 0, 0],
                        [-0.02, 0.0065, 0],
                        [-0.002, 0.05, 0],
                        [0.019, 0.009, 0]], dtype=np.float64)

    # cap.set(3,1280)
    # cap.set(4,720)
    cap.set(3,640)
    cap.set(4,480)
    print(cap.isOpened())

    while True:
        secuess, fream = cap.read()
        if not secuess:
            print("false in reading cam")
            break

        #得到目标形状的像素点坐标  第0个是面积最大点
        points = points_get(fream)
        
        #计算各点到第0点的斜率,排序
        angles = []
        if len(points) == 4:
            for i in range(4):
                if i !=0:
                    angles.append([i,angle(points[0][0],points[0][1],points[i][0],points[i][1])])
                    # angles[i][1] = angle(circles[0][index_0][0],circles[0][index_0][1],circles[0][i][0],circles[0][i][1])
                    # cv2.putText(image_circle, str(angle_i), (int(circles[0][i][0]), int(circles[0][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            b = sorted(angles, key=lambda angles : angles[1])
            print('b',b)
            if b[-1][1]-b[0][1]>180:
                for i in range(len(b)):
                    if b[i][1]<180:
                        b[i][1]+=360
            
            c = sorted(b, key=lambda angles : angles[1])
            # c[i][0] 表示该点在points里的索引,c[i][1]表示该点斜率,从小到大排序
            # b [[3, 22.60133238412697], [2, 120.1507552240974], [1, 339.40467801888485]]
            # c [[1, 339.40467801888485], [3, 382.601332384127], [2, 480.1507552240974]]
            print('c',c)
            for i in range(4):
                if i !=0:
                    for j in range(len(c)):
                        if c[j][0]==i:
                            cv2.putText(fream, str(int(c[j][1])), (int(points[i][0]), int(points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
            imgPoints_circle = np.array([   [points[c[1][0]][0], points[c[1][0]][1]],   [points[c[0][0]][0], points[c[0][0]][1]],                                
                                               [points[0][0], points[0][1]], [points[c[2][0]][0], points[c[2][0]][1]]], dtype=np.float64)
            
            objPoints 
            retval,rvecpnp,tvecpnp  = cv2.solvePnP(objPoints, imgPoints_circle, K, dist)
            aruco.drawAxis( fream, K, dist,rvecpnp[:, :], tvecpnp[:, :], 0.02)

            print('+++++++++++++++++=')
    
        
        
        # gray = cv2.cvtColor(fream, cv2.COLOR_BGR2GRAY)
        # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        # parameters =  aruco.DetectorParameters_create()
        # #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
        # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
        
        
        # # cv2.Rodrigues()

        
        
        # if ids is not None:
        #     a = list(corners)
        #     print('corners',a[0])
        #     print('corners',a[0][0])
        #     print('corners',np.array(a[0][0][0]))
        #     print('corners',a[0][0][0][0])
        #     rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, K, dist)
        #     # 估计每个标记的姿态并返回值rvet和tvec ---不同
        #     print('QR',rvec,tvec)
        #     (rvec-tvec).any() 
        #     if len(corners):
        #         imgPoints = np.array([[a[0][0][0][0], a[0][0][0][1]], [a[0][0][1][0], a[0][0][1][1]], [a[0][0][2][0], a[0][0][2][1]], [a[0][0][3][0], a[0][0][3][1]]], dtype=np.float64)
        #         retval,rvecpnp,tvecpnp  = cv2.solvePnP(objPoints, imgPoints, K, None)
        #         print('PnP',retval,rvecpnp,tvecpnp)
        #     for i in range(rvec.shape[0]):
        #         aruco.drawAxis( fream, K, dist, rvecpnp, tvecpnp, 0.03)
        #         aruco.drawDetectedMarkers(fream, corners)
  
        # main_(fream1)
        cv2.imshow("img_", fream)

        if cv2.waitKey(1)  == ord('q'):
            break




    cap.release()
    cv2.destroyAllWindows()

main()