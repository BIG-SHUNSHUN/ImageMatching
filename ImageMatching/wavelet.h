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
		src : ԭͼ�񣬽������ײ�ͼ��
		layers : ���������������µ��ϵı������Ϊ [0, 1, 2, 3, 4, ... ]�����һ����Դͼ��
		name : С����
		*/
		WaveletPyramid(Mat src, int layers, string name);

		~WaveletPyramid();

		void ShowImage(int layerIdx, const char * type);

		/*
		��ȡС���任�Ľ��

		layerIdx : ��ı��
		type : 'A'��ʾ��ȡ����ͼ��'H'��ʾˮƽ����, 'V'��ʾ��ֱ����, 'D'��ʾ�ԽǷ�������

		WvCoeffs : ��_data��_rows��_cols������Ա����������
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