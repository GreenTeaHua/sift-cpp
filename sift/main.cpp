/*************************************************************************************************************************
*文件说明:
*        SIFT算法的实现
*开发环境:
*        Win10+VS2012+OpenCv2.4.8
*时间地点:
*        陕西师范大学.文津楼 2016.12.30
*再次修改时间:
*        陕西师范大学.文津楼 2017.2.24
*作者信息:
*        九月, opencv3.4.3 win10 fift提取ok，by syh 2018.10.30
**************************************************************************************************************************/
#include <windows.h>
#include <iostream>
#include <vector>
#include "sift.h"


using namespace std;
using namespace cv;


int main(int argc, char **argv)
{
	cv::Mat src = imread("d:/1.jpg");

	if (src.empty())
	{
		cout << "jobs_512.jpg open error! " << endl;
		getchar();
		return -1;
	}

	if (src.channels() != 3) return -2;

	vector<Keypoint> features;

	Sift(src, features, 1.6);                           //【1】SIFT特征点检测和特征点描述
	DrawKeyPoints(src, features);                       //【2】画出关键点(特征点)
	DrawSiftFeatures(src, features);                    //【3】画出SIFT特征点
	write_features(features, "descriptor.txt");         //【4】将特征点写入文本

	cv::imshow("src", src);
	cv::waitKey();

	return 0;
}
