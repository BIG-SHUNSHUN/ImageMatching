#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace shun
{
	enum GAMMA_FORMAT
	{
		INT,
		FLOAT,
		DOUBLE
	};

	struct GammaImagePara
	{
		int lines;    // rows
		int samples;    // cols
		GAMMA_FORMAT format;
	};

	class GammaImage
	{
	public:
		GammaImage();
		~GammaImage();

		void Read(const std::string& dirParams, const std::string& dirData);
		void Show();

	private:
		GammaImagePara _params;
		cv::Mat _img;
		cv::Mat _imgToShow;

		GAMMA_FORMAT Format(std::string str);
		void ReadParams(const std::string& fileName);
		void ReadBinary(const std::string& fileName);
	};

	std::vector<cv::Point> ReadPersistentScatterPoint(const std::string& str);

	void DrawPersistentPoints(cv::Mat image, const std::vector<cv::Point>& points);
}