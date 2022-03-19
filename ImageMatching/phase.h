#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

#define ELEM_TYPE CV_64FC1

class PhaseCongruency
{
	struct EO
	{
		cv::Mat e;
		cv::Mat o;
	};

public:
	PhaseCongruency(int nScale = 4, int nOrient = 6);
	~PhaseCongruency() {}

    void Calc(cv::Mat src);
    void Feature(cv::Mat& outEdges, cv::Mat& outCorners);
	void Orientation(cv::Mat& orient);

	int scaleSize() { return _nScale; }
	int orientSize() { return _nOrient; }
	const std::vector<std::vector<EO>>& eo() { return _eo; }
	cv::Mat pcSum() { return _pcSum; }

private:
    std::vector<cv::Mat> _filter;
	cv::Size _size;    

	std::vector<std::vector<EO>> _eo;   //s*o
	std::vector<cv::Mat> _pc;
	cv::Mat _pcSum;

	cv::Mat _M;
	cv::Mat _m;
	cv::Mat _orient;

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