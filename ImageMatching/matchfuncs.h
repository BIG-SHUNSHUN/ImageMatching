#pragma once

#include <opencv2/opencv.hpp>
#include "phase.h"

using namespace cv;

namespace shun
{
	/*
	ʹ��ģ��ƥ��ķ���ƥ�䵥����

	imgL : ��Ӱ��
	imgR ����Ӱ�� 
	pts : Ҫƥ��ĵ�
	tempSize : ģ�崰�ڴ�С��һ��������
	method : �����Բ�ȷ�����һ��ʹ�ô�NORM�ķ����������Թ�����һ���Ĳ�����
			TM_SQDIFF����ƽ���Ͳ��
			TM_SQDIFF_NORMED����һ����ƽ���Ͳ�ȣ�Ҳ���ǳ��Դ��ڵĻҶ�ƽ����
			TM_CCORR����ػ����
			TM_CCORR_NORMED����һ����ػ���ȣ�Ҳ���ǳ��Դ��ڵĻҶ�ƽ����
			TM_CCOEFF��Э������
			TM_CCOEFF_NORMED����һ��Э�����ȣ�Ҳ�������ϵ����
	*/
	Point TemplateMatchingForPoint(Mat imgL, Mat imgR, const Point& pt, int tempSize, TemplateMatchModes method);

	Mat MatchingUsingSIFT(Mat left, Mat right);

	class RIFT
	{
	public:
		RIFT(int nScale = 4, int nOrient = 6);
		RIFT(const PhaseCongruency& pc);
		RIFT(const RIFT& other) = delete;
		RIFT& operator=(const RIFT& other) = delete;

		void DetectAndCompute(Mat imgIn, std::vector<KeyPoint>& keyPoints, Mat& descriptors);

	private:
		PhaseCongruency _pc;
		int _patchSize = 96;
		int _ns = 6;
		int _ptsNum = 1000;

		void DetectFeature(Mat imgIn, std::vector<KeyPoint>& keyPoints);
		Mat BuildMIM(std::vector<Mat>& CS);
	};

	class Matcher
	{
	public:


	private:

	};

}