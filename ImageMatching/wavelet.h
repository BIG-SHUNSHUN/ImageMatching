#pragma once

#include "../include/wavelib.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace shun
{

	/* 小波影像金字塔

	depends：
		wavelib，refer to github repository：https://github.com/rafat/wavelib

	usage：
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
			src : 原图像，金字塔底层图像
			layers : 金字塔层数，从下到上的标号依次为 [0, 1, 2, 3, 4, ... ]
			name : 小波基

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
		显示数据

		@params 
			layerIdx：层的编号
			type : 'A'表示获取近似图像，'H'表示水平特征, 'V'表示垂直特征, 'D'表示对角方向特征
		*/
		void Show(int layerIdx, const char * type);

		/*
		获取小波变换的结果

		@params
			layerIdx : 层的编号
			type : 'A'表示获取近似图像，'H'表示水平特征, 'V'表示垂直特征, 'D'表示对角方向特征
		*/
		Mat GetCoeffs(int layerIdx, const char* type);

	private:
		vector<wave_object> _wvObjects;
		vector<wt2_object> _wtObjects;
		vector<double*> _wtCoeffs;
		Mat _bottom;    // 最底层图像

		int _layers;
		std::string _name;

		void InitializeByWavelib(Mat src, string name);
		void Release();    // 释放存储
	};

	void WavelibDemo();

	/* haar小波金字塔分解

	  layers：金字塔层数，最底层是原始图像
	*/
	void WaveletDecomposition(Mat img, int layers);

	/* haar小波金字塔重建

	  layers：金字塔层数
	  curLayer：当前层（顶层为1，到底层依次递增）
	  highFreqFactor：高频信息的比例
	*/
	void WaveletRecomstruction(Mat img, int layers, int curLayer, double highFreqFactor);

	void LeastSquareMatch(Mat imgL, Mat imgR, int wsizex, int wsizey, Point2f& ptL, Point2f& ptR);
}