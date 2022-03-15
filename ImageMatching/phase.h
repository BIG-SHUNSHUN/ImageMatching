#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

using EO = std::vector<std::vector<cv::Mat> >;
using PC = std::vector<cv::Mat>;

#define ELEM_TYPE CV_64FC1

class PhaseCongruency
{
public:
	PhaseCongruency(int nScale = 4, int nOrient = 6);
	~PhaseCongruency() {}
    void Calc(cv::Mat src);
    void Feature(cv::Mat& outEdges, cv::Mat& outCorners);

	std::vector<std::vector<cv::Mat> > _eo;   //s*o
	std::vector<cv::Mat> _pc;
	cv::Mat _pcSum;
	cv::Mat _M;
	cv::Mat _m;

	int GetnOrient() { return _nOrient;	};
	int GetnScale() { return _nScale; };

private:
    std::vector<cv::Mat> _filter;
	cv::Size _size;    

	// ²ÎÊý
	int _nOrient = 6;
	int _nScale = 4;
	double _minWavelength = 3.0;
	double _mult = 2.1;
	double _sigmaOnf = 0.55;
	double _k = 2.0;
	double _cutOff = 0.5;
	double _g = 10.0;
	int _noiseMethod = -1;
	double _epsilon = 0.0001;

	void Initialize(cv::Size size);
};