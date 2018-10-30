#pragma once
/*************************************************************************************************************************
*�ļ�˵��:
*        SIFT�㷨��ʵ��ͷ�ļ�
*��������:
*        Win10+VS2012+OpenCv2.4.8
*ʱ��ص�:
*        ����ʦ����ѧ.�Ľ�¥ 2016.12.30
*�ٴ��޸�ʱ��:
*        ����ʦ����ѧ.�Ľ�¥ 2017.2.24
*������Ϣ:
*        ����
**************************************************************************************************************************/
#ifndef SIFT_H
#define SIFT_H
#include<fstream>
#include<iostream>
#include <vector>
#include <myopencv3.h>

using namespace std;
using namespace cv;

typedef double pixel_t;                             //��1����������

#define INIT_SIGMA 0.5                               //��2����ʼsigma
#define SIGMA 1.6
#define INTERVALS 3                                  //��3����˹��������ÿ��ͼ����������/��ͼƬ

#define RATIO 10                                     //��4���뾶r
#define MAX_INTERPOLATION_STEPS 5                    //��5�����ռ���
#define DXTHRESHOLD 0.03                             //��6��|D(x^)| < 0.03   0.03��ֵ��̫��

#define ORI_HIST_BINS 36                             //��7��bings=36
#define ORI_SIGMA_TIMES 1.5
#define ORI_WINDOW_RADIUS 3.0 * ORI_SIGMA_TIMES 
#define ORI_SMOOTH_TIMES 2
#define ORI_PEAK_RATIO 0.8
#define FEATURE_ELEMENT_LENGTH 128
#define DESCR_HIST_BINS 8
#define IMG_BORDER 5 
#define DESCR_WINDOW_WIDTH 4
#define DESCR_SCALE_ADJUST 3
#define DESCR_MAG_THR 0.2
#define INT_DESCR_FCTR 512.0
/*********************************************************************************************
*ģ��˵����
*        �ؼ���/������Ľṹ������
*ע���1��
*        �ڸ�˹�����������Ĺ����У�һ��ͼ����Բ����ü���(octave)ͼ��һ��ͼ���������(inteval)
*        ͼ��
*********************************************************************************************/
struct Keypoint
{
	int    octave;                                        //��1���ؼ���������
	int    interval;                                      //��2���ؼ������ڲ�
	double offset_interval;                               //��3��������Ĳ������

	int    x;                                             //��4��x,y����,����octave��interval��ȡ�Ĳ���ͼ��
	int    y;
	double scale;                                         //��5���ռ�߶�����scale = sigma0*pow(2.0, o+s/S)

	double dx;                                            //��6�����������꣬�����걻���ų�ԭͼ���С 
	double dy;

	double offset_x;
	double offset_y;

	//============================================================
	//1---��˹���������ڸ���߶����꣬��ͬ�����ͬ���sigmaֵ��ͬ
	//2---�ؼ�������������ڳ߶�
	//============================================================
	double octave_scale;                                  //��1��offset_i;
	double ori;                                           //��2������
	int    descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH];            //��3��������������            
	double val;                                           //��4����ֵ
};
/*********************************************************************************
*ģ��˵����
*        SIFT�㷨��,���г�Ա����������
*********************************************************************************/
void read_features(vector<Keypoint>&features, const char*file);
void write_features(const vector<Keypoint>&features, const char*file);

void testInverse3D();

void write_pyr(const vector<Mat>& pyr, const char* dir);
void DrawKeyPoints(Mat &src, vector<Keypoint>& keypoints);

const char* GetFileName(const char* dir, int i);

void ConvertToGray(const Mat& src, Mat& dst);
void DownSample(const Mat& src, Mat& dst);
void UpSample(const Mat& src, Mat& dst);

void GaussianTemplateSmooth(const Mat &src, Mat &dst, double);
void GaussianSmooth2D(const Mat &src, Mat &dst, double sigma);
void GaussianSmooth(const Mat &src, Mat &dst, double sigma);

void Sift(const Mat &src, vector<Keypoint>& features, double sigma = SIGMA, int intervals = INTERVALS);

void CreateInitSmoothGray(const Mat &src, Mat &dst, double);
void GaussianPyramid(const Mat &src, vector<Mat>&gauss_pyr, int octaves, int intervals, double sigma);

void Sub(const Mat& a, const Mat& b, Mat & c);

void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals);
void DetectionLocalExtrema(const vector<Mat>& dog_pyr, vector<Keypoint>& extrema, int octaves, int intervals);
void DrawSiftFeatures(Mat& src, vector<Keypoint>& features);

#endif
