#include "features.h"
#include <corecrt_math_defines.h>

using namespace cv;
using namespace std;

int shun::HarrisDetector::DetectAndCompute(const Mat & img, std::vector<Point2f>& pts, int radius)
{
	Mat harris;
	cornerHarris(img, harris, _blockSize, _kernelSize, _k, BORDER_CONSTANT);

	//Mat normalized;
	//normalize(harris, normalized, 0, 255, NORM_MINMAX, CV_8U);
	//double minVal, maxVal;
	//minMaxLoc(normalized, &minVal, &maxVal);
	//uchar thresh = saturate_cast<uchar>((1 - _threshRatio) * maxVal + _threshRatio * minVal);
	//Mat betterThanThresh = normalized > thresh;
	
	// 极大值抑制，窗口为5 * 5
	Mat dilated;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(harris, dilated, kernel);
	Mat nonMaxSuppression = dilated == harris;

	//Mat result;
	//bitwise_and(betterThanThresh, nonMaxSuppression, result);

	int rows = nonMaxSuppression.rows;
	int cols = nonMaxSuppression.cols;
	int count = 0;
	for (int i = 0; i < rows; i++)
	{
		uchar* ptr = nonMaxSuppression.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			if (ptr[j])
			{
				float x1 = j - radius;
				float x2 = j + radius;
				float y1 = i - radius;
				float y2 = i + radius;

				if (x1 < 0 || x2 >= cols || y1 < 0 || y2 >= rows)
					continue;

				pts.push_back(Point2f(j, i));
				count++;
			}
		}
	}

	return count++;
}

void shun::DrawFeaturePoints(Mat img, const std::vector<Point2f>& pts)
{
	assert(img.type() == CV_8UC3);
	for (int i = 0; i < pts.size(); i++)
	{
		circle(img, pts[i], 3, Scalar(0, 0, 255), -1);
	}
}

void shun::ShiTomashiDetector::DetectAndCompute(const Mat & img, std::vector<Point2f>& pts)
{
	goodFeaturesToTrack(img, pts, _maxCorners, _qualityLevel, _minDistance, Mat(), _blockSize, _useHarris, _k);
}

shun::NonlinearSpace::NonlinearSpace(int nLayer, double scaleValue, DIFFUSION_FUNCTION whichDiff, double sigma1, double sigma2, double ratio, double perc)
	: _nLayer(nLayer), _scaleValue(scaleValue), _whichDiff(whichDiff),
	  _sigma1(sigma1), _sigma2(sigma2), _ratio(ratio), _perc(perc),
	  _space(nLayer)
{
}

void shun::NonlinearSpace::Generate(Mat imgIn)
{
	assert(imgIn.type() == CV_8UC1);

	// 将图像归一化到0——1区间
	Mat img;
	normalize(imgIn, img, 0, 1, NORM_MINMAX, CV_64FC1);

	// 首先对输入图像进行平滑
	int winSize = 2 * round(2 * _sigma1) + 1;
	GaussianBlur(img, img, Size(winSize, winSize), _sigma1, _sigma1, BORDER_REPLICATE);
	_space[0] = img;

	vector<double> sigmas(_nLayer);
	for (int i = 0; i < _nLayer; i++)
	{
		sigmas[i] = _sigma1 * pow(_ratio, i);
	}

	for (int i = 1; i < _nLayer; i++)
	{
		Mat last;
		resize(_space[i - 1], last, Size(), 1 / _scaleValue, 1 / _scaleValue);

		int winSize = 2 * round(2 * _sigma2) + 1;
		Mat lastBlurred;
		GaussianBlur(last, lastBlurred, Size(winSize, winSize), _sigma2, _sigma2, BORDER_REPLICATE);

		Mat lx, ly;
		Sobel(lastBlurred, lx, CV_64FC1, 1, 0, 3, 1, 0, BORDER_REPLICATE);
		Sobel(lastBlurred, ly, CV_64FC1, 0, 1, 3, 1, 0, BORDER_REPLICATE);

		double kPerc = K_PercentileValue(lx, ly, _perc);
		Mat diff;
		switch (_whichDiff)
		{
		case shun::NonlinearSpace::G1:
			diff = PM_G1(lx, ly, kPerc);
			break;
		case shun::NonlinearSpace::G2:
			diff = PM_G2(lx, ly, kPerc);
			break;
		case shun::NonlinearSpace::G3:
			diff = PM_G3(lx, ly, kPerc);
			break;
		default:
			break;
		}

		double step = 1.0 / 2 * (sigmas[i] * sigmas[i] - sigmas[i - 1] * sigmas[i - 1]);
		_space[i] = AOS(last, step, diff);
	}
}

Mat shun::NonlinearSpace::GetLayer(int i)
{
	assert(i >= 0 && i < _nLayer);
	return _space[i];
}

cv::Mat shun::NonlinearSpace::PM_G1(cv::Mat lx, cv::Mat ly, double k)
{
	Mat g1;
	cv::exp(-(lx.mul(lx) + ly.mul(ly)) / (k * k), g1);
	return g1;
}

cv::Mat shun::NonlinearSpace::PM_G2(cv::Mat lx, cv::Mat ly, double k)
{
	Mat g2 = 1.0 / (1.0  + (lx.mul(lx) + ly.mul(ly)) / (k * k));
	return g2;
}

cv::Mat shun::NonlinearSpace::PM_G3(cv::Mat lx, cv::Mat ly, double k)
{
	Mat g3;
	cv::pow(lx.mul(lx) + ly.mul(ly), 4, g3);
	cv::exp(-3.315 / (g3 / pow(k, 8)), g3);
	return 1 - g3;
}

double shun::NonlinearSpace::K_PercentileValue(cv::Mat lx, cv::Mat ly, double perc)
{
	double unit = 0.005;
	int rows = lx.rows;
	int cols = lx.cols;
	Mat mag;
	cv::magnitude(lx, ly, mag);
	Mat tempMag(mag, Rect(1, 1, cols - 2, rows - 2));

	vector<double> vals;
	double maxVal = -1, minVal = DBL_MAX;
	for (int i = 0; i < tempMag.rows; i++)
	{
		double* ptr = tempMag.ptr<double>(i);
		for (int j = 0; j < tempMag.cols; j++)
		{
			if (ptr[j] > 0)
			{
				vals.push_back(ptr[j]);
				if (ptr[j] > maxVal)
					maxVal = ptr[j];
				if (ptr[j] < minVal)
					minVal = ptr[j];
			}
		}
	}
	int nbin = round((maxVal - minVal) / unit);
	vector<int> hist(nbin + 1, 0);
	for (int i = 0; i < vals.size(); i++)
	{
		int index = round((vals[i] - minVal) / unit);
		hist[index] += 1;
	}

	int threshold = _perc * vals.size();
	int count = 0;
	int index = -1;
	for (int i = 0; i < nbin; i++)
	{
		count += hist[i];
		if (count >= threshold)
		{
			index = i;
			break;
		}
	}
	double kPerc = index * unit + minVal;
	return kPerc;
}

cv::Mat shun::NonlinearSpace::AOS(cv::Mat last, double step, cv::Mat diff)
{
	Mat row = AOS_row(last, step, diff);
	Mat col = AOS_col(last, step, diff);
	return 1.0 / 2.0 * (row + col);
}

cv::Mat shun::NonlinearSpace::AOS_row(cv::Mat last, double step, cv::Mat diff)
{
	int rows = last.rows;
	int cols = last.cols;
	Mat U1 = Mat::zeros(last.size(), CV_64FC1);
	Mat a, b, c, d, x;
	for (int r = 0; r < rows; r++)
	{
		d = last.row(r);

		diff.row(r).copyTo(a);
		a(Range::all(), Range(1, cols - 1)) = 2 * a(Range::all(), Range(1, cols - 1));
		a(Range::all(), Range(0, cols - 1)) = a(Range::all(), Range(0, cols - 1)) + 
			                                  diff.row(r)(Range::all(), Range(1, cols));
		a(Range::all(), Range(1, cols)) = a(Range::all(), Range(1, cols)) +
			diff.row(r)(Range::all(), Range(0, cols - 1));
		a = -1.0 / 2.0 * a;

		b = diff.row(r)(Range::all(), Range(0, cols - 1)) + 
			    diff.row(r)(Range::all(), Range(1, cols));
		b = 1.0 / 2.0 * b;

		b.copyTo(c);

		a = 1 - 2 * step * a;
		b = -2 * step * b;
		c = -2 * step * c;

		x = Thomas_Algorithm(a, b, c, d);
		x.copyTo(U1.row(r));
	}
	return U1;
}

cv::Mat shun::NonlinearSpace::AOS_col(cv::Mat last, double step, cv::Mat diff)
{
	int rows = last.rows;
	int cols = last.cols;
	Mat U2 = Mat::zeros(last.size(), CV_64FC1);
	Mat a, b, cc, d, x;
	for (int c = 0; c < cols; c++)
	{
		d = last.col(c);

		diff.col(c).copyTo(a);
		a(Range(1, rows - 1), Range::all()) = 2 * a(Range(1, rows - 1), Range::all());
		a(Range(0, rows - 1), Range::all()) = a(Range(0, rows - 1), Range::all()) +
			diff.col(c)(Range(1, rows), Range::all());
		a(Range(1, rows), Range::all()) = a(Range(1, rows), Range::all()) +
			diff.col(c)(Range(0, rows - 1), Range::all());
		a = -1.0 / 2.0 * a;

		b = diff.col(c)(Range(0, rows - 1), Range::all()) +
			diff.col(c)(Range(1, rows), Range::all());
		b = 1.0 / 2.0 * b;

		b.copyTo(cc);

		a = 1 - 2 * step * a;
		b = -2 * step * b;
		cc = -2 * step * cc;

		x = Thomas_Algorithm(a.t(), b.t(), cc.t(), d.t());
		Mat(x.t()).copyTo(U2.col(c));
	}
	return U2;
}

cv::Mat shun::NonlinearSpace::Thomas_Algorithm(cv::Mat a, cv::Mat b, cv::Mat c, cv::Mat d)
{
	// a, b, c, d 都是横向量

	int N = a.cols;

	vector<double> m(N, 0);
	vector<double> L(N - 1, 0);
	m[0] = a.ptr<double>(0)[0];
	for (int i = 0; i < N - 1; i++)
	{
		L[i] = c.ptr<double>(0)[i] / m[i];
		m[i + 1] = a.ptr<double>(0)[i + 1] - L[i] * b.ptr<double>(0)[i];
	}

	vector<double> y(N, 0);
	y[0] = d.ptr<double>(0)[0];
	for (int i = 1; i < N; i++)
	{
		y[i] = d.ptr<double>(0)[i] - L[i - 1] * y[i - 1];
	}

	Mat x = Mat::zeros(1, N, CV_64FC1);
	x.ptr<double>(0)[N - 1] = y[N - 1] / m[N - 1];
	for (int i = N - 2; i >= 0; i--)
	{
		x.ptr<double>(0)[i] = (y[i] - b.ptr<double>(0)[i] * x.ptr<double>(0)[i + 1]) / m[i];
	}
	return x;
}

//void shun::HAPCG::DetectAndCompute(cv::Mat imgIn, std::vector<cv::Point2f>& keypoints, cv::Mat & descriptors, std::vector<int>& layerBelongTo)
//{
//	_space.Generate(imgIn);
//	InformationFromPhaseCongruency();
//	DetectHarrisCorner(keypoints, layerBelongTo);
//
//	Mat polarRadius = Mat::zeros(2 * _blockRadius.back() + 1, 2 * _blockRadius.back() + 1, CV_32FC1);
//	Mat	polarAngle = Mat::zeros(2 * _blockRadius.back() + 1, 2 * _blockRadius.back() + 1, CV_32FC1);
//	for (int r = 0; r < 2 * _blockRadius.back() + 1; r++)
//	{
//		float* ptrRadius = polarRadius.ptr<float>(r);
//		float* ptrAngle = polarAngle.ptr<float>(r);
//		for (int c = 0; c < 2 * _blockRadius.back() + 1; c++)
//		{
//			int dx = c - _blockRadius.back();
//			int dy = r - _blockRadius.back();
//
//			float r = sqrt(dx * dx + dy * dy);
//			float angle = atan2(dy, dx);
//
//			ptrRadius[c] = r;
//			ptrAngle[c] = angle;
//		}
//	}
//	
//	descriptors.create(keypoints.size(), 200, CV_32FC1);
//	for (int i = 0; i < keypoints.size(); i++)
//	{
//		float x = keypoints[i].x;
//		float y = keypoints[i].y;
//		int layer = layerBelongTo[i];
//
//		float x1 = x - _blockRadius.back() / pow(2.0, i);
//		float x2 = x + _blockRadius.back() / pow(2.0, i);
//		float y1 = y - _blockRadius.back() / pow(2.0, i);
//		float y2 = y + _blockRadius.back() / pow(2.0, i);
//
//		Mat desBlock = Mat::zeros(1, 200, CV_32FC1);
//
//		Mat gradPatch(_pc[layer], Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
//		Mat anglePacth(_angle[layer], Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
//		Mat desPatch = Histgram(gradPatch, anglePacth, Mat(), Mat(), 0, _blockRadius.back / pow(2.0, i));
//
//		desPatch.copyTo(Mat(desBlock, Rect(0, 0, 8, 1)));
//
//		for (int k = 1; k < _blockRadius.size(); k++)
//		{
//			int radiusCur = _blockRadius[k];
//			int radiusLast = _blockRadius[k - 1];
//			float dAngle = 2 * M_PI / 8;
//			for (int o = 0; o < 8; o++)
//			{
//				vector<int> xList;
//				vector<int> yList;
//				int x1 = x + radiusLast * cos(o * dAngle);
//				int y1 = y + radiusLast * sin(o * dAngle);
//				xList.push_back(x1);
//				yList.push_back(y1);
//				int x2 = x + radiusLast * cos((o + 1) * dAngle);
//				int y2 = y + radiusLast * sin((o + 1) * dAngle);
//				xList.push_back(x2);
//				yList.push_back(y2);
//				int x3 = x + radiusCur * cos(o * dAngle);
//				int y3 = y + radiusCur * sin(o * dAngle);
//				xList.push_back(x3);
//				yList.push_back(y3);
//				int x4 = x + radiusCur * cos((o + 1) * dAngle);
//				int y4 = y + radiusCur * sin((o + 1) * dAngle);
//				xList.push_back(x4);
//				yList.push_back(y4);
//
//				sort(xList.begin(), xList.end());
//				sort(yList.begin(), yList.end());
//				int xMin = xList[0], xMax = xList.back();
//				int yMin = yList[0], yMax = yList.back();
//
//				Rect roi = Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1);
//				Mat gradPatch(_pc[i], roi);
//				Mat anglePatch(_angle[i], roi);
//				Mat polarRadiusPatch(polarRadius, roi);    // TODO
//				Mat polarAnglePatch(polarAngle, roi);    // TODO
//
//				Mat despPatch = Histgram(gradPatch, anglePatch, polarRadiusPatch, polarAnglePatch, radiusCur, radiusLast, o * dAngle, (o + 1) * dAngle);
//				desPatch.copyTo(Mat(desBlock, Rect((k * 8 + o) * 8 + 8, 0, 8, 1)));
//			}
//		}
//
//		descriptors.row(i) = desBlock;
//	}
//}
//
//void shun::HAPCG::InformationFromPhaseCongruency()
//{
//	int nLayer = _space.size();
//	for (int i = 0; i < nLayer; i++)
//	{
//		PhaseCongruency pc;
//		pc.Calc(_space.GetLayer(i));
//
//		Mat M, m;
//		pc.Feature(M, m);
//
//		Mat W = (M + m + _deltaPhi * (M - m)) / 2.0;
//		_W.push_back(W);
//
//		_pc.push_back(pc.pcSum());
//
//		// TODO phase congruency angle
//	}
//}
//
//void shun::HAPCG::DetectHarrisCorner(std::vector<cv::Point2f>& keypoints, std::vector<int>& layerBelongTo)
//{
//	int n = _space.size();
//	HarrisDetector harris;
//	for (int i = 0; i < _space.size(); i++)
//	{
//		int count = harris.DetectAndCompute(_W[i], keypoints, _blockRadius[0] / pow(2.0, i));
//
//		for (int i = 0; i < count; i++)
//		{
//			layerBelongTo.push_back(i);
//		}
//	}
//}

bool CompareFunc(const KeyPoint& lhs, const KeyPoint& rhs)
{
	return lhs.response > rhs.response;
}

shun::RIFT::RIFT(int nScale, int nOrient)
	: _pc(nScale, nOrient)
{
}

shun::RIFT::RIFT(const PhaseCongruency & pc)
	: _pc(pc)
{
}

void shun::RIFT::DetectAndCompute(Mat imgIn, vector<KeyPoint>& keyPoints, Mat& descriptors)
{
	// 提取特征点
	DetectFeature(imgIn, keyPoints);

	// 构建MIM
	int nOrient = _pc.orientSize();
	int nScale = _pc.scaleSize();
	vector<Mat> CS(nOrient);
	for (int o = 0; o < nOrient; o++)
	{
		Mat tmp = Mat::zeros(imgIn.size(), CV_64FC1);
		for (int s = 0; s < nScale; s++)
		{
			Mat tempE = _pc.eo()[s][o].e;
			Mat tempO = _pc.eo()[s][o].o;

			Mat mag;
			magnitude(tempE, tempO, mag);
			tmp = tmp + mag;
		}
		CS[o] = tmp;
	}
	Mat MIM = BuildMIM(CS);

	// 为每一个点建立描述向量（每一列代表一个描述向量）
	descriptors.create(_ns * _ns * nOrient, keyPoints.size(), CV_32FC1);
	for (int i = 0; i < keyPoints.size(); i++)
	{
		int x = keyPoints[i].pt.x;
		int y = keyPoints[i].pt.y;

		int x1 = x - _patchSize / 2;
		int y1 = y - _patchSize / 2;
		int x2 = x + _patchSize / 2;
		int y2 = y + _patchSize / 2;

		Mat patch(MIM, Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
		int ys = patch.rows;
		int xs = patch.cols;
		Mat RIFT_des = Mat::zeros(_ns * _ns * nOrient, 1, CV_32FC1);
		for (int j = 0; j < _ns; j++)    // 分块为_ns * _ns个小块，对每个小块计算直方图，然后首尾拼起来
		{
			for (int k = 0; k < _ns; k++)
			{
				double step = (double)ys / _ns;
				int yc1 = round(j * step);
				int yc2 = round((j + 1) * step);
				int xc1 = round(k * step);
				int xc2 = round((k + 1) * step);

				Mat clip(patch, Rect(xc1, yc1, xc2 - xc1, yc2 - yc1));

				// 建立直方图
				Mat hist;
				float ranges[] = { 1, _ns + 1 };
				const float* histRange = { ranges };
				calcHist(&clip, 1, 0, Mat(), hist, 1, &_ns, &histRange);

				Mat roi(RIFT_des, Rect(0, nOrient * (j * _ns + k), 1, nOrient));
				hist.copyTo(roi);
			}
		}

		// 归一化
		double normVal = norm(RIFT_des, NORM_L2);
		if (normVal != 0)
			RIFT_des = RIFT_des / normVal;

		RIFT_des.copyTo(descriptors.col(i));
	}
}

bool PtsCompare(const KeyPoint& lhs, const KeyPoint& rhs)
{
	return lhs.response > rhs.response;
}

void shun::RIFT::DetectFeature(Mat imgIn, vector<KeyPoint>& keyPoints)
{
	_pc.Calc(imgIn);

	Mat M, m;
	_pc.Feature(M, m);

	// 归一化
	// OpenCV的Fast特征检测算子只接受CV_8UC1的灰度图
	normalize(M, M, 0, 255, NORM_MINMAX);
	M.convertTo(M, CV_8UC1);

	vector<KeyPoint> pts;
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(13);
	detector->detect(M, pts);

	sort(pts.begin(), pts.end(), PtsCompare);    // 排序，以便选择角点响应值最强的点

	for (int i = 0; i < pts.size() && i < _ptsNum; i++)
	{
		int x = pts[i].pt.x;
		int y = pts[i].pt.y;

		int x1 = x - _patchSize / 2;
		int y1 = y - _patchSize / 2;
		int x2 = x + _patchSize / 2;
		int y2 = y + _patchSize / 2;

		if (x1 < 0 || y1 < 0 || x2 >= imgIn.cols || y2 >= imgIn.rows)    // 如果超限
		{
			continue;
		}
		else
		{
			keyPoints.push_back(pts[i]);
		}
	}
}

Mat shun::RIFT::BuildMIM(vector<Mat>& CS)
{
	int rows = CS[0].rows;
	int cols = CS[0].cols;

	Mat MIM = Mat::zeros(rows, cols, CV_8UC1);
	for (int r = 0; r < rows; r++)
	{
		uchar* ptrMIM = MIM.ptr<uchar>(r);
		for (int c = 0; c < cols; c++)
		{
			int iMax = 0;
			double maxVal = CS[0].at<double>(r, c);
			for (int o = 1; o < CS.size(); o++)
			{
				double val = CS[o].at<double>(r, c);
				if (val > maxVal)
				{
					iMax = o;
					maxVal = val;
				}
			}
			ptrMIM[c] = iMax + 1;
		}
	}

	return MIM;
}