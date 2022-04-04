#pragma once

#include "../include/wavelib.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace shun
{

	/* С��Ӱ�������

	depends��
		wavelib��refer to github repository��https://github.com/rafat/wavelib

	usage��
		Mat src = imread("xxxxxx");
		WaveletPyramid pyd(3, "harr");
		pyd.build(src);

		Mat src = imread("xxxxxx");
		WaveletPyramid pyd;
		pyd.build(src, 3, "harr");
	*/
	class WaveletPyramid
	{
	public:

		/*
		@ params
			src : ԭͼ�񣬽������ײ�ͼ��
			layers : ���������������µ��ϵı������Ϊ [0, 1, 2, 3, 4, ... ]
			name : С����

		Available Wavelets
			Haar : haar

			Daubechies : db1,db2,.., ,db15

			Biorthogonal : bior1.1 ,bior1.3 ,bior1.5 ,bior2.2 ,bior2.4 ,bior2.6 ,
			               bior2.8 ,bior3.1 ,bior3.3 ,bior3.5 ,bior3.7 ,bior3.9 ,
						   bior4.4 ,bior5.5 ,bior6.8

			Coiflets : coif1,coif2,coif3,coif4,coif5

			Symmlets: sym2,........, sym10 ( Also known as Daubechies' least 
			          asymmetric orthogonal wavelets and represented by the alphanumeric la )
		*/
		WaveletPyramid() {}
		WaveletPyramid(int layers, string name);
		WaveletPyramid(Mat src, int layers, string name);
		~WaveletPyramid();

		void Build(Mat src, int layers, string name);
		void Build(Mat src);

		/*
		��ʾ����

		@params 
			layerIdx����ı��
			type : 'A'��ʾ��ȡ����ͼ��'H'��ʾˮƽ����, 'V'��ʾ��ֱ����, 'D'��ʾ�ԽǷ�������
		*/
		void Show(int layerIdx, const char * type);

		/*
		��ȡС���任�Ľ��

		@params
			layerIdx : ��ı��
			type : 'A'��ʾ��ȡ����ͼ��'H'��ʾˮƽ����, 'V'��ʾ��ֱ����, 'D'��ʾ�ԽǷ�������
		*/
		Mat GetCoeffs(int layerIdx, const char* type);

	private:
		vector<wave_object> _wvObjects;
		vector<wt2_object> _wtObjects;
		vector<double*> _wtCoeffs;
		Mat _bottom;    // ��ײ�ͼ��

		int _layers;
		std::string _name;

		void InitializeByWavelib(Mat src, string name);
		void Release();    // �ͷŴ洢
	};

	void WavelibDemo();

	/* haarС���������ֽ�

	  layers����������������ײ���ԭʼͼ��
	*/
	void WaveletDecomposition(Mat img, int layers);

	/* haarС���������ؽ�

	  layers������������
	  curLayer����ǰ�㣨����Ϊ1�����ײ����ε�����
	  highFreqFactor����Ƶ��Ϣ�ı���
	*/
	void WaveletRecomstruction(Mat img, int layers, int curLayer, double highFreqFactor);

	void LeastSquareMatch(Mat imgL, Mat imgR, int wsizex, int wsizey, Point2f& ptL, Point2f& ptR);
}