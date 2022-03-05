#pragma once

#include "../include/wavelib.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace shun
{

	struct WvCoeffs
	{
		double* _data;
		int _rows;
		int _cols;

		WvCoeffs(double* data, int r, int c)
			: _data(data), _rows(r), _cols(r) {}

		WvCoeffs() {}
	};

	class WaveletPyramid
	{
	public:

		/*
		src : 原图像，金字塔底层图像
		layers : 金字塔层数，从下到上的标号依次为 [0, 1, 2, 3, 4, ... ]，最后一层是源图像
		name : 小波基
		*/
		WaveletPyramid(Mat src, int layers, string name);

		~WaveletPyramid();

		void ShowImage(int layerIdx, const char * type);

		/*
		获取小波变换的结果

		layerIdx : 层的编号
		type : 'A'表示获取近似图像，'H'表示水平特征, 'V'表示垂直特征, 'D'表示对角方向特征

		WvCoeffs : 有_data、_rows、_cols三个成员，意义自明
		*/
		Mat GetCoeffs(int layerIdx, const char* type);

	private:
		vector<wave_object> _wvObjects;
		vector<wt2_object> _wtObjects;
		vector<double*> _wtCoeffs;
		Mat _bottom;
		int _layers;

		void InitializeByWavelib(Mat src, string name);
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