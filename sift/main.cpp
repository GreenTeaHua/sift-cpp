/*************************************************************************************************************************
*�ļ�˵��:
*        SIFT�㷨��ʵ��
*��������:
*        Win10+VS2012+OpenCv2.4.8
*ʱ��ص�:
*        ����ʦ����ѧ.�Ľ�¥ 2016.12.30
*�ٴ��޸�ʱ��:
*        ����ʦ����ѧ.�Ľ�¥ 2017.2.24
*������Ϣ:
*        ����, opencv3.4.3 win10 fift��ȡok��by syh 2018.10.30
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

	Sift(src, features, 1.6);                           //��1��SIFT�������������������
	DrawKeyPoints(src, features);                       //��2�������ؼ���(������)
	DrawSiftFeatures(src, features);                    //��3������SIFT������
	write_features(features, "descriptor.txt");         //��4����������д���ı�

	cv::imshow("src", src);
	cv::waitKey();

	return 0;
}
