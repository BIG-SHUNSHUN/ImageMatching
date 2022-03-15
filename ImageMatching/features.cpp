#include "features.h"

using namespace cv;
using namespace std;

void shun::HarrisDetector::DetectAndCompute(const Mat & img, std::vector<Point>& pts)
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
	for (int i = 0; i < rows; i++)
	{
		uchar* ptr = nonMaxSuppression.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			if (ptr[j])
				pts.push_back(Point(j, i));
		}
	}
}

void shun::DrawFeaturePoints(Mat img, const std::vector<Point>& pts)
{
	assert(img.type() == CV_8UC3);
	for (int i = 0; i < pts.size(); i++)
	{
		circle(img, pts[i], 3, Scalar(0, 0, 255), -1);
	}
}

void shun::ShiTomashiDetector::DetectAndCompute(const Mat & img, std::vector<Point>& pts)
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
