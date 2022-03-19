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



	Mat FastSampleConsensus(std::vector<Point2f> small1, 
		                    std::vector<Point2f> small2,
		                    std::vector<Point2f> large1,
		                    std::vector<Point2f> large2,
		                    int iters);
}