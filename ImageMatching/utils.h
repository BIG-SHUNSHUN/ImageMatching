#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

namespace shun
{
	/* ɾ���ַ�����β��ch�ַ���Ĭ��Ϊ�ո�
	*/
	void Trim(string & s, char ch = ' ');

	/* ����ch���ַ���s�и�
	*/
	void Split(vector<string>& splitString, string&s, char ch);

	inline uint32_t reversebytes_uint32t(uint32_t value) {
		return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 |
			(value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24;
	}

	inline uint16_t reversebytes_uint16(uint16_t value)
	{
		return (value & 0x00FFU) << 8 | (value & 0xFF00U) >> 8;
	}

	/* ����Ҷ�任Ƶ�Ƽ���������
	*/
	void FFTShift(Mat& inAndOut);
	void IFFTShift(Mat& inAndOut);
	void CircShift(Mat& inAndOut, Size step);

	void MeshGrid(Mat x, Mat y, Mat& matX, Mat& matY);

	void CountTime(void(*func)());

	double ValuePercent(Mat in, double prec);

	void Atan2ForMat(cv::Mat inX, cv::Mat inY, cv::Mat& out);
}