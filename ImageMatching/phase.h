#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

using EO = std::vector<std::vector<cv::Mat> >;
using PC = std::vector<cv::Mat>;

class PhaseCongruency
{
public:
	PhaseCongruency(int nScale = 4, int nOrient = 6);
	~PhaseCongruency() {}
    void Calc(cv::Mat src);
    void Feature(cv::Mat& outEdges, cv::Mat& outCorners);

	const EO& GetEO() const { return _eo; }
	const PC& GetPC() const { return _pc; }
    const cv::Mat& GetPcSum() const { return _pcSum; }

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

private:
    std::vector<cv::Mat> _filter;
	cv::Size _size;    

	std::vector<std::vector<cv::Mat> > _eo;   //s*o
	std::vector<cv::Mat> _pc;
	cv::Mat _pcSum;
	cv::Mat _M;
	cv::Mat _m;

	void Initialize(cv::Size size);
};