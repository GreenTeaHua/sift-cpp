/*************************************************************************************************************************
*�ļ�˵��:
*        SIFT�㷨��ͷ�ļ�����Ӧ��ʵ���ļ�
*��������:
*        Win10+VS2012+OpenCv2.4.8
*ʱ��ص�:
*        ����ʦ����ѧ.�Ľ�¥ 2016.12.30
*�ٴ��޸�ʱ��:
*        ����ʦ����ѧ.�Ľ�¥ 2017.2.24
*������Ϣ:
*        ����
**************************************************************************************************************************/
#include"sift.h"
#include<fstream>
#include<iostream>
#include <vector>
using namespace std;
using namespace cv;
/*************************************************************************************************************************
*ģ��˵����
*        ͼ��ҶȻ�����----����ɫͼ��תΪ�Ҷ�ͼ��
**************************************************************************************************************************/
void ConvertToGray(const Mat& src, Mat& dst)
{
	cv::Size size = src.size();
	if (dst.empty())
		dst.create(size, CV_64F);                              //[1]����Mat��ĳ�Ա��������Mat����

	uchar*    srcData = src.data;                             //[2]ָ��洢��������ֵ�ľ������������
	pixel_t*  dstData = (pixel_t*)dst.data;                   //[3]ָ��dst�ľ�������������

	int       dstStep = dst.step / sizeof(dstData[0]);         //[4]һ��ͼ���ж����ֽ���

	for (int j = 0; j<src.cols; j++)
	{
		for (int i = 0; i<src.rows; i++)
		{
			double b = (srcData + src.step*i + src.channels()*j)[0] / 255.0;
			double g = (srcData + src.step*i + src.channels()*j)[1] / 255.0;
			double r = (srcData + src.step*i + src.channels()*j)[2] / 255.0;
			*(dstData + dstStep*i + dst.channels()*j) = (r + g + b) / 3.0;
		}
	}
}
/*************************************************************************************************************************
*ģ��˵����
*        �������
**************************************************************************************************************************/
void DownSample(const Mat& src, Mat& dst)
{
	if (src.channels() != 1)
		return;

	if (src.cols <= 1 || src.rows <= 1)
	{
		src.copyTo(dst);
		return;
	}

	dst.create((int)(src.rows / 2), (int)(src.cols / 2), src.type());

	pixel_t* srcData = (pixel_t*)src.data;
	pixel_t* dstData = (pixel_t*)dst.data;

	int srcStep = src.step / sizeof(srcData[0]);
	int dstStep = dst.step / sizeof(dstData[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols; j += 2, n++)
	{
		m = 0;
		for (int i = 0; i < src.rows; i += 2, m++)
		{
			pixel_t sample = *(srcData + srcStep * i + src.channels() * j);
			if (m < dst.rows && n < dst.cols)
			{
				*(dstData + dstStep * m + dst.channels() * n) = sample;
			}
		}
	}

}
/*************************************************************************************************************************
*ģ��˵����
*        ���Բ�ֵ�Ŵ�
**************************************************************************************************************************/
void UpSample(const Mat &src, Mat &dst)
{
	if (src.channels() != 1)
		return;
	dst.create(src.rows * 2, src.cols * 2, src.type());

	pixel_t* srcData = (pixel_t*)src.data;
	pixel_t* dstData = (pixel_t*)dst.data;

	int srcStep = src.step / sizeof(srcData[0]);
	int dstStep = dst.step / sizeof(dstData[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols - 1; j++, n += 2)
	{
		m = 0;
		for (int i = 0; i < src.rows - 1; i++, m += 2)
		{
			double sample = *(srcData + srcStep * i + src.channels() * j);
			*(dstData + dstStep * m + dst.channels() * n) = sample;

			double rs = *(srcData + srcStep * (i)+src.channels()*j) + (*(srcData + srcStep * (i + 1) + src.channels()*j));
			*(dstData + dstStep * (m + 1) + dst.channels() * n) = rs / 2;
			double cs = *(srcData + srcStep * i + src.channels()*(j)) + (*(srcData + srcStep * i + src.channels()*(j + 1)));
			*(dstData + dstStep * m + dst.channels() * (n + 1)) = cs / 2;

			double center = (*(srcData + srcStep * (i + 1) + src.channels() * j))
				+ (*(srcData + srcStep * i + src.channels() * j))
				+ (*(srcData + srcStep * (i + 1) + src.channels() * (j + 1)))
				+ (*(srcData + srcStep * i + src.channels() * (j + 1)));

			*(dstData + dstStep * (m + 1) + dst.channels() * (n + 1)) = center / 4;

		}

	}



	if (dst.rows < 3 || dst.cols < 3)
		return;

	//�����������
	for (int k = dst.rows - 1; k >= 0; k--)
	{
		*(dstData + dstStep *(k)+dst.channels()*(dst.cols - 2)) = *(dstData + dstStep *(k)+dst.channels()*(dst.cols - 3));
		*(dstData + dstStep *(k)+dst.channels()*(dst.cols - 1)) = *(dstData + dstStep *(k)+dst.channels()*(dst.cols - 3));
	}
	for (int k = dst.cols - 1; k >= 0; k--)
	{
		*(dstData + dstStep *(dst.rows - 2) + dst.channels()*(k)) = *(dstData + dstStep *(dst.rows - 3) + dst.channels()*(k));
		*(dstData + dstStep *(dst.rows - 1) + dst.channels()*(k)) = *(dstData + dstStep *(dst.rows - 3) + dst.channels()*(k));
	}

}

/*************************************************************************************************************************
*ģ��˵����
*        OpenCv�еĸ�˹ƽ������
**************************************************************************************************************************/
void GaussianSmooth(const Mat &src, Mat &dst, double sigma)
{
	GaussianBlur(src, dst, Size(0, 0), sigma, sigma);
}
/*************************************************************************************************************************
*ģ��˵����
*        ģ��1--������ʼ�Ҷ�ͼ��------��ʼͼ���Ƚ�ԭͼ��ҶȻ���������һ����ʹ�ø�˹ģ��ƽ��
*����˵����
*        ���ʼ������˹��������ʱ��ҪԤ��ģ�������ͼ������Ϊ��0����ĵ�0��ͼ����ʱ�൱�ڶ�������ߵĿ���Ĳ����ʡ���
*        �ˣ�ͨ�����������Ƚ�ͼ��ĳ߶�����һ�������ɵ�-1�顣���Ǽٶ���ʼ������ͼ��Ϊ�˿������������Ѿ����������sigma=0.5
*        �ĸ�˹ģ�����������ͼ��ĳߴ���˫���Բ�ֵ����һ������ô�൱��sigma=1.0
*��������ԭ��:
*        �ڼ�⼫ֵ��ǰ����ԭʼͼ��ĸ�˹ƽ������ͼ��ʧ��Ƶ��Ϣ������Lowe�����ڽ����߶ȿռ�ǰ�����ȶ�ԭʼͼ�񳤿���չһ����
*        �Ա㱣��ԭʼͼ�����Ϣ(�ر���ͼ��ĸ�Ƶ��Ϣ�������Ե���ǵ�)�������������������
*��������:
*        ������������������ڴ�����˹��������-1��ͼ��
*������:
*        2017.2.24----����ʦ����ѧ
**************************************************************************************************************************/
void CreateInitSmoothGray(const Mat &src, Mat &dst, double sigma = SIGMA)
{
	cv::Mat gray;                                  //[1]����ԭʼͼ��ĻҶ�ͼ��        
	cv::Mat up;                                    //[2]����ԭʼͼ����ϲ���ͼ��

	ConvertToGray(src, gray);
	UpSample(gray, up);                            //[3]ͼ����ϲ���
												   //[4]��˹��������-1���sigma�ļ��㹫ʽ
	double  sigma_init = sqrt(sigma * sigma - (INIT_SIGMA * 2) * (INIT_SIGMA * 2));//[1]-1���sigma

	GaussianSmooth(up, dst, sigma_init);           //[5]��˹ƽ��
}
/*************************************************************************************************************************
*ģ��˵����
*        ģ�����3.3 ͼ���˹�������Ĺ���
**************************************************************************************************************************/
void GaussianPyramid(const Mat &src, vector<Mat>&gauss_pyr, int octaves, int intervals = INTERVALS, double sigma = SIGMA)
{
	double *sigmas = new double[intervals + 3];
	double k = pow(2.0, 1.0 / intervals);

	sigmas[0] = sigma;

	double sig_prev;
	double sig_total;

	for (int i = 1; i < intervals + 3; i++)
	{
		sig_prev = pow(k, i - 1) * sigma;
		sig_total = sig_prev * k;
		sigmas[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);
	}

	for (int o = 0; o < octaves; o++)
	{
		//ÿ�������
		for (int i = 0; i < intervals + 3; i++)
		{
			Mat mat;
			if (o == 0 && i == 0)
			{
				src.copyTo(mat);
			}
			else if (i == 0)
			{
				DownSample(gauss_pyr[(o - 1)*(intervals + 3) + intervals], mat);
			}
			else
			{
				GaussianSmooth(gauss_pyr[o * (intervals + 3) + i - 1], mat, sigmas[i]);
			}
			gauss_pyr.push_back(mat);
		}
	}

	delete[] sigmas;
}
/*************************************************************************************************************************
*ģ��˵����
*        ͼ��Ĳ��
**************************************************************************************************************************/
void Sub(const Mat& a, const Mat& b, Mat & c)
{
	if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type())
		return;
	if (!c.empty())
		return;
	c.create(a.size(), a.type());

	pixel_t* ap = (pixel_t*)a.data;
	pixel_t* ap_end = (pixel_t*)a.dataend;
	pixel_t* bp = (pixel_t*)b.data;
	pixel_t* cp = (pixel_t*)c.data;
	int step = a.step / sizeof(pixel_t);

	while (ap != ap_end)
	{
		*cp++ = *ap++ - *bp++;
	}
}
/*************************************************************************************************************************
*ģ��˵����
*       ģ������3.4 ��˹��ֽ�����
*����˵����
*       1--2002�꣬Mikolajczyk����ϸ��ʵ��Ƚ��з��ֳ߶ȹ�һ���ĸ�˹������˹�����ļ���ֵ�ͼ�Сֵͬ������������ȡ���������磺
*          �ݶȣ�Hessian����Harris�ǵ������Ƚϣ��ܹ������ȶ���ͼ��������
*       2--��Lindberg����1994�귢�ָ�˹��ֺ�����Difference of Gaussian,���DOG���ӣ���߶ȹ�һ����������˹�����ǳ����ƣ���ˣ�
*          ����ͼ���Ĳ�ִ���΢�֡�
**************************************************************************************************************************/
void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals = INTERVALS)
{
	for (int o = 0; o < octaves; o++)
	{
		for (int i = 1; i < intervals + 3; i++)
		{
			Mat mat;
			Sub(gauss_pyr[o*(intervals + 3) + i], gauss_pyr[o*(intervals + 3) + i - 1], mat);
			dog_pyr.push_back(mat);
		}
	}
}
/*************************************************************************************************************************
*ģ��˵����
*       ģ���ĵĵ�һ��:3.4-�ռ伫ֵ��ĳ������(�ؼ���ĳ���̽��)
*����˵��:
*       1--�ڸ�˹��ֺ���֮�󣬵õ�һϵ�еĹؼ�������Ƶ㣬������Ҫ����Щ�ؼ�������Ƶ�������м���ɸѡ
*       2--�˿���������ݵ�ԭ��ΪCSDN�����еģ�3.5�ռ伫ֵ��ļ��
**************************************************************************************************************************/
bool isExtremum(int x, int y, const vector<Mat>& dog_pyr, int index)
{
	pixel_t * data = (pixel_t *)dog_pyr[index].data;
	int      step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t   val = *(data + y*step + x);

	if (val > 0)
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(data[0]);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					if (val < *((pixel_t*)dog_pyr[index + i].data + stp*(y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	else
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(pixel_t);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					//�����С��ֵ
					if (val > *((pixel_t*)dog_pyr[index + i].data + stp*(y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	return true;
}
/*************************************************************************************************************************
*ģ��˵����
*       ģ���ĵĵ�����:4.2--������Ե��Ӧ��
*����˵��:
*       1��һ�����岻�õĸ�˹������ӵļ�ֵ�ں���Ե�ĵط��нϴ��ס�����ʣ��ڴ�ֱ��Ե�ķ����н�С�������ʡ�
*       2��DOG���ӻ������ǿ�ı�Ե��Ӧ����Ҫ�޳����ȶ��ı�Ե��Ӧ�㣬��ȡ�����㴦��Hessian����������ͨ��һ��2*2��Hessian��
*          ��H���
*       3��������D��Hessian���������ֵ�����ȣ���ʽ(r+1)*(r+1)/r��ֵ����������ֵ���ʱ��С�����ֵԽ��˵����������ֵ�ı�ֵ
*          Խ�󣬼���ĳһ��������ݶ�ֵԽ�󣬶�����һ��������ݶ�ֵԽС������Եǡǡ����������������ԣ�Ϊ���޳���Ե��Ӧ�㣬
*          ��Ҫ�øñ�ֵС��һ������ֵ����ˣ�Ϊ�˼���������Ƿ���ĳ��ֵr�£�ֻ���⡣CSDN�����еĹ�ʽ(4-7����)������������
*          ���㱣������֮���޳���
**************************************************************************************************************************/
#define DAt(x, y) (*(data+(y)*step+(x))) 
bool passEdgeResponse(int x, int y, const vector<Mat>& dog_pyr, int index, double r = RATIO)
{
	pixel_t *data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y*step + x);

	double Dxx;
	double Dyy;
	double Dxy;
	double Tr_h;                                                         //[1]Hessian����ļ�
	double Det_h;                                                        //[2]Hessian��������Ӧ������ʽ��ֵ

	Dxx = DAt(x + 1, y) + DAt(x - 1, y) - 2 * val;
	Dyy = DAt(x, y + 1) + DAt(x, y - 1) - 2 * val;
	Dxy = (DAt(x + 1, y + 1) + DAt(x - 1, y - 1) - DAt(x - 1, y + 1) - DAt(x + 1, y - 1)) / 4.0;

	Tr_h = Dxx + Dyy;
	Det_h = Dxx * Dyy - Dxy * Dxy;

	if (Det_h <= 0)return false;

	if (Tr_h * Tr_h / Det_h < (r + 1) * (r + 1) / r) return true;

	return false;
}
/*************************************************************************************************************************
*ģ��˵����
*       ���޲���󵼣�
**************************************************************************************************************************/
#define Hat(i, j) (*(H+(i)*3 + (j)))

double PyrAt(const vector<Mat>& pyr, int index, int x, int y)
{
	pixel_t *data = (pixel_t*)pyr[index].data;
	int      step = pyr[index].step / sizeof(data[0]);
	pixel_t   val = *(data + y*step + x);

	return val;
}
/*************************************************************************************************************************
*ģ��˵����
*       ���޲���󵼣�
**************************************************************************************************************************/
#define At(index, x, y) (PyrAt(dog_pyr, (index), (x), (y)))

//3άD(x)һ��ƫ��,dx������
void DerivativeOf3D(int x, int y, const vector<Mat>& dog_pyr, int index, double *dx)
{
	double Dx = (At(index, x + 1, y) - At(index, x - 1, y)) / 2.0;
	double Dy = (At(index, x, y + 1) - At(index, x, y - 1)) / 2.0;
	double Ds = (At(index + 1, x, y) - At(index - 1, x, y)) / 2.0;

	dx[0] = Dx;
	dx[1] = Dy;
	dx[2] = Ds;
}

//3άD(x)����ƫ������Hessian����
void Hessian3D(int x, int y, const vector<Mat>& dog_pyr, int index, double *H)
{
	double val, Dxx, Dyy, Dss, Dxy, Dxs, Dys;

	val = At(index, x, y);

	Dxx = At(index, x + 1, y) + At(index, x - 1, y) - 2 * val;
	Dyy = At(index, x, y + 1) + At(index, x, y - 1) - 2 * val;
	Dss = At(index + 1, x, y) + At(index - 1, x, y) - 2 * val;

	Dxy = (At(index, x + 1, y + 1) + At(index, x - 1, y - 1)
		- At(index, x + 1, y - 1) - At(index, x - 1, y + 1)) / 4.0;

	Dxs = (At(index + 1, x + 1, y) + At(index - 1, x - 1, y)
		- At(index - 1, x + 1, y) - At(index + 1, x - 1, y)) / 4.0;

	Dys = (At(index + 1, x, y + 1) + At(index - 1, x, y - 1)
		- At(index + 1, x, y - 1) - At(index - 1, x, y + 1)) / 4.0;

	Hat(0, 0) = Dxx;
	Hat(1, 1) = Dyy;
	Hat(2, 2) = Dss;

	Hat(1, 0) = Hat(0, 1) = Dxy;
	Hat(2, 0) = Hat(0, 2) = Dxs;
	Hat(2, 1) = Hat(1, 2) = Dys;
}
/*************************************************************************************************************************
*ģ��˵����
*       4.4 ���׾�������
**************************************************************************************************************************/
#define HIat(i, j) (*(H_inve+(i)*3 + (j)))
//3*3�׾�������
bool Inverse3D(const double *H, double *H_inve)
{

	double A = Hat(0, 0)*Hat(1, 1)*Hat(2, 2)
		+ Hat(0, 1)*Hat(1, 2)*Hat(2, 0)
		+ Hat(0, 2)*Hat(1, 0)*Hat(2, 1)
		- Hat(0, 0)*Hat(1, 2)*Hat(2, 1)
		- Hat(0, 1)*Hat(1, 0)*Hat(2, 2)
		- Hat(0, 2)*Hat(1, 1)*Hat(2, 0);

	if (fabs(A) < 1e-10) return false;

	HIat(0, 0) = Hat(1, 1) * Hat(2, 2) - Hat(2, 1)*Hat(1, 2);
	HIat(0, 1) = -(Hat(0, 1) * Hat(2, 2) - Hat(2, 1) * Hat(0, 2));
	HIat(0, 2) = Hat(0, 1) * Hat(1, 2) - Hat(0, 2)*Hat(1, 1);

	HIat(1, 0) = Hat(1, 2) * Hat(2, 0) - Hat(2, 2)*Hat(1, 0);
	HIat(1, 1) = -(Hat(0, 2) * Hat(2, 0) - Hat(0, 0) * Hat(2, 2));
	HIat(1, 2) = Hat(0, 2) * Hat(1, 0) - Hat(0, 0)*Hat(1, 2);

	HIat(2, 0) = Hat(1, 0) * Hat(2, 1) - Hat(1, 1)*Hat(2, 0);
	HIat(2, 1) = -(Hat(0, 0) * Hat(2, 1) - Hat(0, 1) * Hat(2, 0));
	HIat(2, 2) = Hat(0, 0) * Hat(1, 1) - Hat(0, 1)*Hat(1, 0);

	for (int i = 0; i < 9; i++)
	{
		*(H_inve + i) /= A;
	}
	return true;
}
/*************************************************************************************************************************
*ģ��˵����
*
**************************************************************************************************************************/
//����x^
void GetOffsetX(int x, int y, const vector<Mat>& dog_pyr, int index, double *offset_x)
{
	//x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
	double H[9], H_inve[9] = { 0 };
	Hessian3D(x, y, dog_pyr, index, H);
	Inverse3D(H, H_inve);
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	for (int i = 0; i < 3; i++)
	{
		offset_x[i] = 0.0;
		for (int j = 0; j < 3; j++)
		{
			offset_x[i] += H_inve[i * 3 + j] * dx[j];
		}
		offset_x[i] = -offset_x[i];
	}
}

//����|D(x^)|
double GetFabsDx(int x, int y, const vector<Mat>& dog_pyr, int index, const double* offset_x)
{
	//|D(x^)|=D + 0.5 * dx * offset_x; dx=(Dx, Dy, Ds)^T
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	double term = 0.0;
	for (int i = 0; i < 3; i++)
		term += dx[i] * offset_x[i];

	pixel_t *data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y*step + x);

	return fabs(val + 0.5 * term);
}
/*************************************************************************************************************************
*ģ��˵����
*       ģ���ĵĵڶ���:������ֵ�㣬ɾ�����ȶ��ĵ�
*����˵��:
*       1--���ݸ�˹��ֺ��������ļ�ֵ�㲢��ȫ�����ȶ��������㣬��ΪĳЩ��ֵ�����Ӧ����������DOG���ӻ������ǿ�ı�Ե��Ӧ
*       2--���Ϸ�����⵽�ļ�ֵ������ɢ�ռ�ļ�ֵ�㣬����ͨ�������ά���κ�������ȷ��λ�ؼ����λ�úͳ߶ȣ�ͬʱȥ���Աȶ�
*          �ͺͲ��ȶ��ı�Ե��Ӧ��(��ΪDOG���ӻ������ǿ�ı�Ե��Ӧ)������ǿƥ����ȶ��ԡ���߿�������������
*       3--������ֵ�㣬ɾ�����ȶ��㣬|D(x)| < 0.03 Lowe 2004
**************************************************************************************************************************/
Keypoint* InterploationExtremum(int x, int y, const vector<Mat>& dog_pyr, int index, int octave, int interval, double dxthreshold = DXTHRESHOLD)
{
	//����x=(x,y,sigma)^T
	//x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
	double offset_x[3] = { 0 };

	const Mat &mat = dog_pyr[index];

	int idx = index;
	int intvl = interval;
	int i = 0;

	while (i < MAX_INTERPOLATION_STEPS)
	{
		GetOffsetX(x, y, dog_pyr, idx, offset_x);
		//4. Accurate keypoint localization.  Lowe
		//���offset_x ����һά�ȴ���0.5��it means that the extremum lies closer to a different sample point.
		if (fabs(offset_x[0]) < 0.5 && fabs(offset_x[1]) < 0.5 && fabs(offset_x[2]) < 0.5)
			break;

		//����Χ�ĵ����
		x += cvRound(offset_x[0]);
		y += cvRound(offset_x[1]);
		interval += cvRound(offset_x[2]);

		idx = index - intvl + interval;
		//�˴���֤����ʱ x+1,y+1��x-1, y-1��Ч
		if (interval < 1 || interval > INTERVALS || x >= mat.cols - 1 || x < 2 || y >= mat.rows - 1 || y < 2)
		{
			return NULL;
		}

		i++;
	}

	//�ܸ�ʧ��
	if (i >= MAX_INTERPOLATION_STEPS)
		return NULL;

	//rejecting unstable extrema
	//|D(x^)| < 0.03ȡ����ֵ
	if (GetFabsDx(x, y, dog_pyr, idx, offset_x) < dxthreshold / INTERVALS)
	{
		return NULL;
	}

	Keypoint *keypoint = new Keypoint;


	keypoint->x = x;
	keypoint->y = y;

	keypoint->offset_x = offset_x[0];
	keypoint->offset_y = offset_x[1];

	keypoint->interval = interval;
	keypoint->offset_interval = offset_x[2];

	keypoint->octave = octave;

	keypoint->dx = (x + offset_x[0])*pow(2.0, octave);
	keypoint->dy = (y + offset_x[1])*pow(2.0, octave);

	return keypoint;
}
/*************************************************************************************************************************
*ģ��˵����
*       ģ���ģ�3.5 �ռ伫ֵ��ļ��(�ؼ���ĳ���̽��)
*����˵����
*       1--�ؼ�������DOG�ռ�ľֲ���ֵ����ɵģ��ؼ���ĳ���̽����ͨ��ͬһ���ڸ�DoG��������ͼ��֮��ıȽ���ɵġ�Ϊ��Ѱ��DoG
*          �����ļ�ֵ�㣬ÿһ�����ص㶼Ҫ�����������ڵĵ�Ƚϣ������Ƿ������ͼ����ͳ߶������ڵĵ����С��
*       2--��Ȼ���������ļ�ֵ�㲢��ȫ�����ȶ��������㣬��ΪĳЩ��ֵ����Ӧ����������DOG���ӻ������ǿ�ı�Ե��Ӧ��
**************************************************************************************************************************/
void DetectionLocalExtrema(const vector<Mat>& dog_pyr, vector<Keypoint>& extrema, int octaves, int intervals = INTERVALS)
{

	double  thresh = 0.5 * DXTHRESHOLD / intervals;

	for (int o = 0; o < octaves; o++)
	{
		//��һ������һ�㼫ֵ����
		for (int i = 1; i < (intervals + 2) - 1; i++)
		{
			int index = o*(intervals + 2) + i;                              //[1]ͼƬ�����Ķ�λ
			pixel_t *data = (pixel_t *)dog_pyr[index].data;                //[2]��ȡͼƬ�ľ�������׵�ַ
			int step = dog_pyr[index].step / sizeof(data[0]);           //[3]˵�������ڴ洢�ռ��еĴ洢�������Կռ�ķ�ʽ��ŵ�


			for (int y = IMG_BORDER; y < dog_pyr[index].rows - IMG_BORDER; y++)
			{
				for (int x = IMG_BORDER; x < dog_pyr[index].cols - IMG_BORDER; x++)
				{
					pixel_t val = *(data + y*step + x);
					if (std::fabs(val) > thresh)                           //[4]�ų���ֵ��С�ĵ�
					{
						if (isExtremum(x, y, dog_pyr, index))                //[5]�ж��Ƿ��Ǽ�ֵ
						{
							Keypoint *extrmum = InterploationExtremum(x, y, dog_pyr, index, o, i);
							if (extrmum)
							{
								if (passEdgeResponse(extrmum->x, extrmum->y, dog_pyr, index))
								{
									extrmum->val = *(data + extrmum->y*step + extrmum->x);
									extrema.push_back(*extrmum);
								}

								delete extrmum;

							}//extrmum
						}//isExtremum
					}//std::fabs
				}//for x
			}//for y

		}
	}
}
/*************************************************************************************************************************
*ģ��˵����
*       ģ���壺
*����˵����
*
**************************************************************************************************************************/
void CalculateScale(vector<Keypoint>& features, double sigma = SIGMA, int intervals = INTERVALS)
{
	double intvl = 0;
	for (int i = 0; i < features.size(); i++)
	{
		intvl = features[i].interval + features[i].offset_interval;
		features[i].scale = sigma * pow(2.0, features[i].octave + intvl / intervals);
		features[i].octave_scale = sigma * pow(2.0, intvl / intervals);
	}

}

//�������ͼ����������
void HalfFeatures(vector<Keypoint>& features)
{
	for (int i = 0; i < features.size(); i++)
	{
		features[i].dx /= 2;
		features[i].dy /= 2;

		features[i].scale /= 2;
	}
}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ����---����2������ؼ�����ݶȺ��ݶȷ���
*����˵����
*        1������ؼ���(x,y)�����ݶȷ�ֵ���ݶȷ���
*        2����������������ݶȷ�ֵ���ݶȷ��򱣴��ڱ���mag��ori��
*********************************************************************************************************************************/
bool CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori)
{
	if (x > 0 && x < gauss.cols - 1 && y > 0 && y < gauss.rows - 1)
	{
		pixel_t *data = (pixel_t*)gauss.data;
		int step = gauss.step / sizeof(*data);

		double dx = *(data + step*y + (x + 1)) - (*(data + step*y + (x - 1)));           //[1]����X�����ϵĲ�ִ���΢��dx
		double dy = *(data + step*(y + 1) + x) - (*(data + step*(y - 1) + x));           //[2]����Y�����ϵĲ�ִ���΢��dy

		mag = sqrt(dx*dx + dy*dy);                                          //[3]����ùؼ�����ݶȷ�ֵ
		ori = atan2(dy, dx);                                                //[4]����ùؼ�����ݶȷ���
		return true;
	}
	else
		return false;
}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ����---����1�������ݶȵķ���ֱ��ͼ
*����˵����
*        1��ֱ��ͼ��ÿ10��Ϊһ��������36������������ķ���ΪΪ���ص���ݶȷ������ĳ��̴������ݶȷ�ֵ��
*        2������Lowe�Ľ��飬ֱ��ͼͳ�Ʋ���3*1.5*sigma
*        3����ֱ��ͼͳ��ʱ��ÿ�����������ص���ø�˹��Ȩ������Lowe�Ľ��飬ģ�����[0.25,0.5,0.25],����������Ȩ����
*��    �ۣ�
*        ͼ��Ĺؼ�������Ϻ�ÿ���ؼ����ӵ��������Ϣ��λ�á��߶ȡ�����ͬʱҲ��ʹ�ؼ���߱�ƽ�ơ����ź���ת������
*********************************************************************************************************************************/
double* CalculateOrientationHistogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma)
{
	double* hist = new double[bins];                           //[1]��̬����һ��double���͵�����
	for (int i = 0; i < bins; i++)                               //[2]����������ʼ��
		*(hist + i) = 0.0;

	double  mag;                                                //[3]�ؼ�����ݶȷ�ֵ                                          
	double  ori;                                                //[4]�ؼ�����ݶȷ���
	double  weight;

	int           bin;
	const double PI2 = 2.0*CV_PI;
	double        econs = -1.0 / (2.0*sigma*sigma);

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			if (CalcGradMagOri(gauss, x + i, y + j, mag, ori))       //[5]����ùؼ�����ݶȷ�ֵ�ͷ���
			{
				weight = exp((i*i + j*j)*econs);
				bin = cvRound(bins * (CV_PI - ori) / PI2);     //[6]��һ��double�е��������������룬����һ�����ε���
				bin = bin < bins ? bin : 0;

				hist[bin] += mag * weight;                      //[7]ͳ���ݶȵķ���ֱ��ͼ
			}
		}
	}

	return hist;
}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ����---����3�����ݶȷ���ֱ��ͼ�����������εĸ�˹ƽ��
*����˵����
*        1����ֱ��ͼͳ��ʱ��ÿ�����������ص���ø�˹��Ȩ������Lowe�Ľ��飬ģ�����[0.25,0.5,0.25],����������Ȩ����
*        2����ֱ��ͼ��������ƽ��
*********************************************************************************************************************************/
void GaussSmoothOriHist(double *hist, int n)
{
	double prev = hist[n - 1];
	double temp;
	double h0 = hist[0];

	for (int i = 0; i < n; i++)
	{
		temp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * (i + 1 >= n ? h0 : hist[i + 1]);//�Է���ֱ��ͼ���и�˹ƽ��
		prev = temp;
	}
}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ����---����4�����㷽��ֱ��ͼ�е�������
*********************************************************************************************************************************/
double DominantDirection(double *hist, int n)
{
	double maxd = hist[0];
	for (int i = 1; i < n; i++)
	{
		if (hist[i] > maxd)                            //��ȡ36�����е�����ֵ
			maxd = hist[i];
	}
	return maxd;
}
void CopyKeypoint(const Keypoint& src, Keypoint& dst)
{
	dst.dx = src.dx;
	dst.dy = src.dy;

	dst.interval = src.interval;
	dst.octave = src.octave;
	dst.octave_scale = src.octave_scale;
	dst.offset_interval = src.offset_interval;

	dst.offset_x = src.offset_x;
	dst.offset_y = src.offset_y;

	dst.ori = src.ori;
	dst.scale = src.scale;
	dst.val = src.val;
	dst.x = src.x;
	dst.y = src.y;
}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ����---����5��������Ӿ�ȷ�Ĺؼ���������----�����ֵ
*����˵����
*        1������ֱ��ͼ�ķ�ֵ������˸�������ķ�����ֱ��ͼ�е����ֵ��Ϊ�ùؼ����������Ϊ����ǿƥ���³���ԣ�ֻ������ֵ������
*           �����ֵ80%�ķ�����Ϊ�Ĺؼ���ĸ�������ˣ�����ͬһ�ݶ�ֵ�ö����ֵ�Ĺؼ���λ�ã�����ͬλ�úͳ߶Ƚ����ж���ؼ��㱻
*           ����������ͬ������15%�Ĺؼ��㱻���������򣬵��ǿ������Ե���߹ؼ�����ȶ��ԡ�
*        2����ʵ�ʱ���У����ǰѸùؼ��㸴�Ƴɶ�ݹؼ��㣬��������ֵ�ֱ𸳸���Щ���ƺ�Ĺؼ���
*        3�����ң���ɢ���ݶ�ֱ��ͼҪ���С���ֵ��ϴ���������ø��Ӿ�ȷ�ķ���Ƕ�ֵ
*********************************************************************************************************************************/
#define Parabola_Interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r))) 
void CalcOriFeatures(const Keypoint& keypoint, vector<Keypoint>& features, const double *hist, int n, double mag_thr)
{
	double  bin;
	double  PI2 = CV_PI * 2.0;
	int l;
	int r;

	for (int i = 0; i < n; i++)
	{
		l = (i == 0) ? n - 1 : i - 1;
		r = (i + 1) % n;

		//hist[i]�Ǽ�ֵ
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			bin = i + Parabola_Interpolate(hist[l], hist[i], hist[r]);
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);

			Keypoint new_key;

			CopyKeypoint(keypoint, new_key);

			new_key.ori = ((PI2 * bin) / n) - CV_PI;
			features.push_back(new_key);
		}
	}
}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ������5 �ؼ��㷽�����
*����˵����
*        1��Ϊ��ʹ������������ת�����ԣ���Ҫ����ͼ��ľֲ�����Ϊÿһ���ؼ������һ����׼����ʹ��ͼ���ݶȵķ�����ȡ�ֲ��ṹ���ȶ�
*           ����
*        2��������DOG�������м������Ĺؼ��㣬�ɼ������ڸ�˹������ͼ��3sigma���򴰿������ص��ݶȺͷ����ݶȺͷ���������
*        3���ݶȵ�ģ�ͷ���������ʾ:
*        4) ����ɹؼ�����ݶȼ����ʹ��ֱ��ͼͳ�����������ص��ݶȺͷ����ݶ�ֱ��ͼ��0~360�ȵķ���Χ��Ϊ36����������ÿ��10�ȣ�
*           ��ͼ5.1��ʾ��ֱ��ͼ�ķ�ֵ��������˹ؼ����������
*********************************************************************************************************************************/
void OrientationAssignment(vector<Keypoint>& extrema, vector<Keypoint>& features, const vector<Mat>& gauss_pyr)
{
	int n = extrema.size();
	double *hist;

	for (int i = 0; i < n; i++)
	{

		hist = CalculateOrientationHistogram(gauss_pyr[extrema[i].octave*(INTERVALS + 3) + extrema[i].interval],
			extrema[i].x, extrema[i].y, ORI_HIST_BINS, cvRound(ORI_WINDOW_RADIUS*extrema[i].octave_scale),
			ORI_SIGMA_TIMES*extrema[i].octave_scale);                             //[1]�����ݶȵķ���ֱ��ͼ

		for (int j = 0; j < ORI_SMOOTH_TIMES; j++)
			GaussSmoothOriHist(hist, ORI_HIST_BINS);                              //[2]�Է���ֱ��ͼ���и�˹ƽ��
		double highest_peak = DominantDirection(hist, ORI_HIST_BINS);            //[3]��ȡ����ֱ��ͼ�еķ�ֵ
																				 //[4]������Ӿ�ȷ�Ĺؼ���������
		CalcOriFeatures(extrema[i], features, hist, ORI_HIST_BINS, highest_peak*ORI_PEAK_RATIO);

		delete[] hist;

	}
}

void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d)
{
	double d_r, d_c, d_o, v_r, v_c, v_o;
	double** row, *h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor(ybin);
	c0 = cvFloor(xbin);
	o0 = cvFloor(obin);
	d_r = ybin - r0;
	d_c = xbin - c0;
	d_o = obin - o0;

	/*
	����ֵ��
	xbin,ybin,obin:���ӵ������Ӵ��ڵ�λ�úͷ���
	�������ӵ㶼������4*4�Ĵ�����
	r0,c0ȡ������xbin��ybin��������
	r0,c0ֻ��ȡ��0,1,2
	xbin,ybin��(-1, 2)
	r0ȡ������xbin��������ʱ��
	r0+0 <= xbin <= r0+1
	mag������[r0,r1]������ֵ
	obinͬ��
	*/

	for (r = 0; r <= 1; r++)
	{
		rb = r0 + r;
		if (rb >= 0 && rb < d)
		{
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			row = hist[rb];
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				if (cb >= 0 && cb < d)
				{
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					h = row[cb];
					for (o = 0; o <= 1; o++)
					{
						ob = (o0 + o) % bins;
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						h[ob] += v_o;
					}
				}
			}
		}
	}


}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ����--����1:���������ӵ�ֱ��ͼ
*����˵����
*
*********************************************************************************************************************************/
double*** CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
	double ***hist = new double**[width];

	for (int i = 0; i < width; i++)
	{
		hist[i] = new double*[width];
		for (int j = 0; j < width; j++)
		{
			hist[i][j] = new double[bins];
		}
	}

	for (int r = 0; r < width; r++)
		for (int c = 0; c < width; c++)
			for (int o = 0; o < bins; o++)
				hist[r][c][o] = 0.0;


	double cos_ori = cos(ori);
	double sin_ori = sin(ori);

	//6.1��˹Ȩֵ��sigma���������ִ��ڿ�ȵ�һ��
	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma*sigma);

	double PI2 = CV_PI * 2;

	double sub_hist_width = DESCR_SCALE_ADJUST * octave_scale;

	//��1�����������������ͼ����������İ뾶
	int    radius = (sub_hist_width*sqrt(2.0)*(width + 1)) / 2.0 + 0.5;    //[1]0.5ȡ��������
	double grad_ori;
	double grad_mag;

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;

			double xbin = rot_x + width / 2 - 0.5;                         //[2]xbin,ybinΪ����4*4�����е��±�ֵ
			double ybin = rot_y + width / 2 - 0.5;

			if (xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
			{
				if (CalcGradMagOri(gauss, x + j, y + i, grad_mag, grad_ori)) //[3]����ؼ�����ݶȷ���
				{
					grad_ori = (CV_PI - grad_ori) - ori;
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					double obin = grad_ori * (bins / PI2);

					double weight = exp(conste*(rot_x*rot_x + rot_y * rot_y));

					InterpHistEntry(hist, xbin, ybin, obin, grad_mag*weight, bins, width);

				}
			}
		}
	}

	return hist;
}

void NormalizeDescr(Keypoint& feat)
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat.descr_length;

	for (i = 0; i < d; i++)
	{
		cur = feat.descriptor[i];
		len_sq += cur*cur;
	}
	len_inv = 1.0 / sqrt(len_sq);
	for (i = 0; i < d; i++)
		feat.descriptor[i] *= len_inv;
}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ����--����2:ֱ��ͼ�������ӵ�ת��
*����˵����
*
*********************************************************************************************************************************/
void HistToDescriptor(double ***hist, int width, int bins, Keypoint& feature)
{
	int int_val, i, r, c, o, k = 0;

	for (r = 0; r < width; r++)
		for (c = 0; c < width; c++)
			for (o = 0; o < bins; o++)
			{
				feature.descriptor[k++] = hist[r][c][o];
			}

	feature.descr_length = k;
	NormalizeDescr(feature);                           //[1]����������������һ��

	for (i = 0; i < k; i++)                           //[2]��������������
		if (feature.descriptor[i] > DESCR_MAG_THR)
			feature.descriptor[i] = DESCR_MAG_THR;

	NormalizeDescr(feature);                           //[3]�����ӽ������һ�εĹ�һ������

	for (i = 0; i < k; i++)                           //[4]�������ȸ����͵�������ת��Ϊ���ε�������
	{
		int_val = INT_DESCR_FCTR * feature.descriptor[i];
		feature.descriptor[i] = min(255, int_val);
	}
}
/********************************************************************************************************************************
*ģ��˵��:
*        ģ����:6 �ؼ�������
*����˵����
*        1��ͨ�����ϲ��裬����һ���ؼ��㣬ӵ��������Ϣ��λ�á��߶ȡ�����
*        2������������Ϊÿ���ؼ��㽨��һ������������һ��������������ؼ�������������ʹ�䲻����ֱ仯���仯��������ա��ӽǱ仯�ȵ�
*        3����������Ӳ��������ؼ��㣬Ҳ�����ؼ�����Χ���乱�׵����ص㣬����������Ӧ���нϸߵĶ����ԣ��Ա�����������ȷ��ƥ�����
*        1��SIFT������----�ǹؼ��������˹ͼ���ݶ�ͳ�ƽ����һ�ֱ�ʾ��
*        2��ͨ���Թؼ�����Χͼ������ֿ飬��������ݶ�ֱ��ͼ�����ɾ��ж����Ե�����
*        3����������Ǹ�����ͼ����Ϣ��һ�ֱ����ͳ��󣬾���Ψһ�ԡ�
*Lowe���ģ�
*    Lowe����������ʹ���ڹؼ���߶ȿռ���4*4�Ĵ����м����8��������ݶ���Ϣ����4*4*8=128ά����������������Ĳ���������ʾ:
*        1)ȷ�����������������ͼ������
*        2������������תΪ�ؼ���ķ�����ȷ����ת�����ԣ���CSDN�����е�ͼ6.2��ʾ����ת���������������������ͨ����ʽ(6-2)����
*        3���������ڵĲ�������䵽��Ӧ�������򣬽��������ڵ��ݶ�ֵ���䵽8�������ϣ�������Ȩֵ
*        4����ֵ����ÿ�����ӵ�˸�������ݶ�
*        5������ͳ�Ƶ�4*4*8=128���ݶ���Ϣ��Ϊ�ùؼ�����������������������γɺ�Ϊ��ȥ�����ձ仯��Ӱ�죬��Ҫ�����ǽ��й�һ������
*           ����ͼ��Ҷ�ֵ����Ư�ƣ�ͼ�������ݶ���������������õ��ģ�����Ҳ��ȥ�����õ�������������ΪH����һ��֮�������ΪL
*        6���������������ޡ������Թ��գ�������Ͷȱ仯�����ĳЩ������ݶ�ֵ���󣬶��Է����Ӱ��΢������ˣ���������ֵ��������һ��
*           ��һ��ȡ0.2���ضϽϴ���ݶ�ֵ��Ȼ���ڽ���һ�ι�һ��������������ļ����ԡ�
*        7����������ĳ߶ȶ���������������������
*        8�����ˣ�SIFT���������������ɡ�
*********************************************************************************************************************************/
void DescriptorRepresentation(vector<Keypoint>& features, const vector<Mat>& gauss_pyr, int bins, int width)
{
	double ***hist;

	for (int i = 0; i < features.size(); i++)
	{                                                                       //[1]���������ӵ�ֱ��ͼ
		hist = CalculateDescrHist(gauss_pyr[features[i].octave*(INTERVALS + 3) + features[i].interval],
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);

		HistToDescriptor(hist, width, bins, features[i]);                   //[2]ֱ��ͼ�������ӵ�ת��

		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < width; k++)
			{
				delete[] hist[j][k];
			}
			delete[] hist[j];
		}
		delete[] hist;
	}
}

bool FeatureCmp(Keypoint& f1, Keypoint& f2)
{
	return f1.scale < f2.scale;
}
/*******************************************************************************************************************
*����˵��:
*        ����ģ��1��SIFT�㷨ģ��
*��������˵��:
*        1---const Mat &src---------------׼���������������ԭʼͼƬ
*        2---vector<Keypoint>& features---�����洢�������Ĺؼ���
*        3---double sigma-----------------sigmaֵ
*        4---int intervals----------------�ؼ������ڵĲ���
********************************************************************************************************************/
void Sift(const Mat &src, vector<Keypoint>& features, double sigma, int intervals)
{
	std::cout << "��Step_one��Create -1 octave gaussian pyramid image" << std::endl;
	cv::Mat          init_gray;
	CreateInitSmoothGray(src, init_gray, sigma);
	int octaves = log((double)min(init_gray.rows, init_gray.cols)) / log(2.0) - 2;             //�����˹�������Ĳ���
	std::cout << "��1��The height and width of init_gray_img = " << init_gray.rows << "*" << init_gray.cols << std::endl;
	std::cout << "��2��The octaves of the gauss pyramid      = " << octaves << std::endl;


	std::cout << "��Step_two��Building gaussian pyramid ..." << std::endl;
	std::vector<Mat> gauss_pyr;
	GaussianPyramid(init_gray, gauss_pyr, octaves, intervals, sigma);
	write_pyr(gauss_pyr, "gausspyramid");


	std::cout << "��Step_three��Building difference of gaussian pyramid..." << std::endl;
	vector<Mat> dog_pyr;
	DogPyramid(gauss_pyr, dog_pyr, octaves, intervals);
	write_pyr(dog_pyr, "dogpyramid");



	std::cout << "��Step_four��Deatecting local extrema..." << std::endl;
	vector<Keypoint> extrema;
	DetectionLocalExtrema(dog_pyr, extrema, octaves, intervals);
	std::cout << "��3��keypoints cout: " << extrema.size() << " --" << std::endl;
	std::cout << "��4��extrema detection finished." << std::endl;
	std::cout << "��5��please look dir gausspyramid, dogpyramid and extrema.txt.--" << endl;



	std::cout << "��Step_five��CalculateScale..." << std::endl;
	CalculateScale(extrema, sigma, intervals);
	HalfFeatures(extrema);



	std::cout << "��Step_six��Orientation assignment..." << std::endl;
	OrientationAssignment(extrema, features, gauss_pyr);
	std::cout << "��6��features count: " << features.size() << std::endl;



	std::cout << "��Step_seven��DescriptorRepresentation..." << std::endl;
	DescriptorRepresentation(features, gauss_pyr, DESCR_HIST_BINS, DESCR_WINDOW_WIDTH);
	sort(features.begin(), features.end(), FeatureCmp);
	cout << "finished." << endl;
}
/*******************************************************************************************************************
*����˵��:
*        ����SIFT������ľ��庯��
********************************************************************************************************************/
void DrawSiftFeature(Mat& src, Keypoint& feat, CvScalar color)
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	CvPoint start, end, h1, h2;

	/* compute points for an arrow scaled and rotated by feat's scl and ori */
	start_x = cvRound(feat.dx);
	start_y = cvRound(feat.dy);
	scl = feat.scale;
	ori = feat.ori;
	len = cvRound(scl * scale);
	hlen = cvRound(scl * hscale);
	blen = len - hlen;
	end_x = cvRound(len *  cos(ori)) + start_x;
	end_y = cvRound(len * -sin(ori)) + start_y;
	h1_x = cvRound(blen *  cos(ori + CV_PI / 18.0)) + start_x;
	h1_y = cvRound(blen * -sin(ori + CV_PI / 18.0)) + start_y;
	h2_x = cvRound(blen *  cos(ori - CV_PI / 18.0)) + start_x;
	h2_y = cvRound(blen * -sin(ori - CV_PI / 18.0)) + start_y;
	start = cvPoint(start_x, start_y);
	end = cvPoint(end_x, end_y);
	h1 = cvPoint(h1_x, h1_y);
	h2 = cvPoint(h2_x, h2_y);

	line(src, start, end, color, 1, 8, 0);
	line(src, end, h1, color, 1, 8, 0);
	line(src, end, h2, color, 1, 8, 0);
}
/*******************************************************************************************************************
*����˵��:
*         ����ģ��3������SIFT������
********************************************************************************************************************/
void DrawSiftFeatures(Mat& src, vector<Keypoint>& features)
{
	CvScalar color = CV_RGB(0, 255, 0);
	for (int i = 0; i < features.size(); i++)
	{
		DrawSiftFeature(src, features[i], color);
	}
}
/*******************************************************************************************************************
*����˵��:
*         ����ģ��2�������ؼ���KeyPoints
********************************************************************************************************************/
void DrawKeyPoints(Mat &src, vector<Keypoint>& keypoints)
{
	int j = 0;
	for (int i = 0; i < keypoints.size(); i++)
	{

		CvScalar color = { 255, 0 ,0 };
		circle(src, Point(keypoints[i].dx, keypoints[i].dy), 3, color);
		j++;
	}
}

const char* GetFileName(const char* dir, int i)
{
	char *name = new char[50];
	sprintf(name, "%s\\%d\.jpg", dir, i);
	return name;
}

void cv64f_to_cv8U(const Mat& src, Mat& dst)
{
	double* data = (double*)src.data;
	int step = src.step / sizeof(*data);

	if (!dst.empty())
		return;
	dst.create(src.size(), CV_8U);

	uchar* dst_data = dst.data;

	for (int i = 0, m = 0; i < src.cols; i++, m++)
	{
		for (int j = 0, n = 0; j < src.rows; j++, n++)
		{
			*(dst_data + dst.step*j + i) = (uchar)(*(data + step*j + i) * 255);
		}
	}
}


//ͨ��ת���󱣴��ͼ�񣬻�ʧ��,��imshow��ʾ����ͼ�����ܴ�
void writecv64f(const char* filename, const Mat& mat)
{
	Mat dst;
	cv64f_to_cv8U(mat, dst);
	imwrite(filename, dst);
}

void write_pyr(const vector<Mat>& pyr, const char* dir)
{
	for (int i = 0; i < pyr.size(); i++)
	{
		writecv64f(GetFileName(dir, i), pyr[i]);
	}
}

void read_features(vector<Keypoint>&features, const char*file)
{
	ifstream in(file);
	int n = 0, dims = 0;
	in >> n >> dims;
	cout << n << " " << dims << endl;
	for (int i = 0; i < n; i++)
	{
		Keypoint key;
		in >> key.dy >> key.dx >> key.scale >> key.ori;
		for (int j = 0; j < dims; j++)
		{
			in >> key.descriptor[j];
		}
		features.push_back(key);
	}
	in.close();
}
/*******************************************************************************************************************
*����˵��:
*         ����ģ��4����������д���ı��ļ�
********************************************************************************************************************/
void write_features(const vector<Keypoint>&features, const char*file)
{
	ofstream dout(file);
	dout << features.size() << " " << FEATURE_ELEMENT_LENGTH << endl;
	for (int i = 0; i < features.size(); i++)
	{
		dout << features[i].dy << " " << features[i].dx << " " << features[i].scale << " " << features[i].ori << endl;
		for (int j = 0; j < FEATURE_ELEMENT_LENGTH; j++)
		{
			if (j % 20 == 0)
				dout << endl;
			dout << features[i].descriptor[j] << " ";
		}
		dout << endl;
	}
	dout.close();
}
