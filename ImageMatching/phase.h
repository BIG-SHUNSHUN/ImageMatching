#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace shun
{
	#define ELEM_TYPE CV_64FC1

	// 相位一致性参数
	struct PhaseCongruencyParams
	{
		int nOrient = 6;
		int nScale = 4;
		double minWavelength = 3.0;
		double mult = 2.1;
		double sigmaOnf = 0.55;
		double k = 2.0;
		double cutOff = 0.5;
		double g = 10.0;
		int noiseMethod = -1;
		double epsilon = 0.0001;
	};

	/* 相位一致性

	reference code：https://peterkovesi.com/matlabfns/#phasecong

	默认参数：
		nOrient = 6;
		nScale = 4;
		minWavelength = 3.0;
		mult = 2.1;
		sigmaOnf = 0.55;
		k = 2.0;
		cutOff = 0.5;
		g = 10.0;
		noiseMethod = -1;
		epsilon = 0.0001;

	usage:
		Mat src = imread("xxxxx");
	    PhaseCongruency pc;    // create phase congruency object
		pc.SetParams(4, 6);    // parameters setting
		pc.Prepare(src);    // do some preparation

		Mat edge, corner;
		pc.Feature(edge, corner);    // compute edge features and corner features

	*/
	class PhaseCongruency
	{
		struct EO
		{
			cv::Mat e;
			cv::Mat o;
		};

	public:
		PhaseCongruency();
		~PhaseCongruency() {}

		void SetParams(const PhaseCongruencyParams& params);
		void SetParams(int nScale = 4, int nOrient = 6, double minWavelength = 3.0, double mult = 2.1,
			double sigmaOnf = 0.55, double k = 2.0, double cutoff = 0.5, double g = 10.0,
			double noiseMethod = -1);
		const PhaseCongruencyParams& Params();

		void Prepare(cv::Mat src);

		void Feature(cv::Mat& outEdges, cv::Mat& outCorners);
		void Orientation(cv::Mat& orient);

		const std::vector<std::vector<EO>>& eo() { return _eo; }
		cv::Mat pcSum() { return _pcSum; }

	private:
		cv::Size _size;
		PhaseCongruencyParams _params;    // 参数

		std::vector<std::vector<EO>> _eo;   // s*o
		std::vector<cv::Mat> _pc;
		cv::Mat _pcSum;

		// 一些辅助函数
		void Helper(int n, std::vector<double>& arr);
		cv::Mat LowPassFilter(cv::Size size, double cutOff, int n);
	};
}