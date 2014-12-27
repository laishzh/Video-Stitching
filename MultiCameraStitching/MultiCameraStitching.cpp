// MultiCameraStitching.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include < stdio.h >
#include <math.h>
#include < opencv2\opencv.hpp >    
#include < opencv2\stitching\stitcher.hpp >  
#include <opencv2/nonfree/nonfree.hpp>
      
#ifdef _DEBUG    
#pragma comment(lib, "opencv_calib3d247d.lib")
#pragma comment(lib, "opencv_contrib247d.lib")
#pragma comment(lib, "opencv_core247d.lib")
#pragma comment(lib, "opencv_features2d247d.lib")
#pragma comment(lib, "opencv_flann247d.lib")
#pragma comment(lib, "opencv_gpu247d.lib")
#pragma comment(lib, "opencv_highgui247d.lib")
#pragma comment(lib, "opencv_imgproc247d.lib")
#pragma comment(lib, "opencv_legacy247d.lib")
#pragma comment(lib, "opencv_ml247d.lib")
#pragma comment(lib, "opencv_nonfree247d.lib")
#pragma comment(lib, "opencv_objdetect247d.lib")
#pragma comment(lib, "opencv_ocl247d.lib")
#pragma comment(lib, "opencv_photo247d.lib")
#pragma comment(lib, "opencv_stitching247d.lib")
#pragma comment(lib, "opencv_superres247d.lib")
#pragma comment(lib, "opencv_ts247d.lib")
#pragma comment(lib, "opencv_video247d.lib")
#pragma comment(lib, "opencv_videostab247d.lib")
#else    
#pragma comment(lib, "opencv_calib3d247.lib")
#pragma comment(lib, "opencv_contrib247.lib")
#pragma comment(lib, "opencv_core247.lib")
#pragma comment(lib, "opencv_features2d247.lib")
#pragma comment(lib, "opencv_flann247.lib")
#pragma comment(lib, "opencv_gpu247.lib")
#pragma comment(lib, "opencv_highgui247.lib")
#pragma comment(lib, "opencv_imgproc247.lib")
#pragma comment(lib, "opencv_legacy247.lib")
#pragma comment(lib, "opencv_ml247.lib")
#pragma comment(lib, "opencv_nonfree247.lib")
#pragma comment(lib, "opencv_objdetect247.lib")
#pragma comment(lib, "opencv_ocl247.lib")
#pragma comment(lib, "opencv_photo247.lib")
#pragma comment(lib, "opencv_stitching247.lib")
#pragma comment(lib, "opencv_superres247.lib")
#pragma comment(lib, "opencv_ts247.lib")
#pragma comment(lib, "opencv_video247.lib")
#pragma comment(lib, "opencv_videostab247.lib")
#endif    
      
using namespace cv;    
using namespace std;  
      

Mat src,frameImg;  
int width;  
int height;  
vector<Point> srcCorner(4);  
vector<Point> dstCorner(4);
long det_x, det_y;

static bool createDetectorDescriptorMatcher( const string& detectorType, const string& descriptorType, const string& matcherType,  
	Ptr<FeatureDetector>& featureDetector,  
	Ptr<DescriptorExtractor>& descriptorExtractor,  
	Ptr<DescriptorMatcher>& descriptorMatcher )  
{  
	cout <<"<Creating feature detector, descriptor extractor and descriptor matcher ..." <<endl;  
	if (detectorType=="SIFT"||detectorType=="SURF")  
		initModule_nonfree();  
	featureDetector = FeatureDetector::create( detectorType );  
	descriptorExtractor = DescriptorExtractor::create( descriptorType );  
	descriptorMatcher = DescriptorMatcher::create( matcherType );  
	cout <<">" <<endl;  
	bool isCreated = !( featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty() );  
	if( !isCreated )  
		cout <<"Can not create feature detector or descriptor extractor or descriptor matcher of given types." <<endl <<">" <<endl;  
	return isCreated;  
}  


bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,    
	const std::vector<cv::KeyPoint>& trainKeypoints,     
	float reprojectionThreshold,    
	std::vector<cv::DMatch>& matches,    
	cv::Mat& homography  )  
{  
	const int minNumberMatchesAllowed = 5;    
	if (matches.size() <minNumberMatchesAllowed)    
		return false;    
	// Prepare data for cv::findHomography    
	std::vector<cv::Point2f> queryPoints(matches.size());    
	std::vector<cv::Point2f> trainPoints(matches.size());    
	for (size_t i = 0; i <matches.size(); i++)    
	{    
		queryPoints[i] = queryKeypoints[matches[i].queryIdx].pt;    
		trainPoints[i] = trainKeypoints[matches[i].trainIdx].pt;    
	}    
	// Find homography matrix and get inliers mask

	std::vector<unsigned char> inliersMask(matches.size());    
	homography = cv::findHomography(queryPoints,     
		trainPoints,     
		CV_FM_RANSAC,     
		reprojectionThreshold,     
		inliersMask);
	for (size_t i = 0; i < homography.rows; ++i)
	{
		for (size_t j = 0; j < homography.cols; ++j)
			printf("%lf ", homography.at<double>(i, j));
		printf("\n");
	}
	std::vector<cv::DMatch> inliers;
	for (size_t i=0; i<inliersMask.size(); i++)    
	{    
		if (inliersMask[i])    
			inliers.push_back(matches[i]);    
	}    
	matches.swap(inliers);  
	Mat homoShow;  
	drawMatches(src,queryKeypoints,frameImg,trainKeypoints,matches,homoShow,Scalar::all(-1),CV_RGB(255,255,255),Mat(),2);
	long sum_x, sum_y;
	sum_x = sum_y = 0;

	for (size_t i = 0; i < matches.size(); ++i) {
		KeyPoint from = queryKeypoints[matches[i].queryIdx];
		KeyPoint to = trainKeypoints[matches[i].trainIdx];

		printf("From: (%f, %f) to (%f, %f).\n", 
			from.pt.x, from.pt.y,
			to.pt.x, to.pt.y);
		sum_x += to.pt.x - from.pt.x;
		sum_y += to.pt.y - from.pt.y;
	}

	det_x = sum_x / matches.size();
	det_y = sum_y / matches.size();
	printf("Det: %ld, %ld.\n", det_x, det_y);
	imshow("homoShow",homoShow);   

	return matches.size() > minNumberMatchesAllowed;   

}  


bool matchingDescriptor(const vector<KeyPoint>& queryKeyPoints,const vector<KeyPoint>& trainKeyPoints,  
	const Mat& queryDescriptors,const Mat& trainDescriptors,   
	Ptr<DescriptorMatcher>& descriptorMatcher,  
	Mat &homo,
	bool enableRatioTest = true)  
{  
	vector<vector<DMatch>> m_knnMatches;  
	vector<DMatch>m_Matches;  

	if (enableRatioTest)  
	{  
		cout<<"KNN Matching"<<endl;  
		const float minRatio = 1.f / 1.5f;  
		descriptorMatcher->knnMatch(queryDescriptors,trainDescriptors,m_knnMatches,2);  
		for (size_t i=0; i<m_knnMatches.size(); i++)  
		{  
			const cv::DMatch& bestMatch = m_knnMatches[i][0];  
			const cv::DMatch& betterMatch = m_knnMatches[i][1];  
			float distanceRatio = bestMatch.distance / betterMatch.distance;  
			if (distanceRatio <minRatio)  
			{
				m_Matches.push_back(bestMatch);  
			}
		}  

	}  
	else  
	{  
		cout<<"Cross-Check"<<endl;  
		Ptr<cv::DescriptorMatcher> BFMatcher(new cv::BFMatcher(cv::NORM_HAMMING, true));  
		BFMatcher->match(queryDescriptors,trainDescriptors, m_Matches );  
	}  
  
	float homographyReprojectionThreshold = 20;  
	bool homographyFound = refineMatchesWithHomography(  
		queryKeyPoints,trainKeyPoints,homographyReprojectionThreshold,m_Matches,homo);  

	if (!homographyFound)  
		return false;  
	else  
	{  
		if (m_Matches.size()>=10)
		{
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( src.cols, 0 );
			obj_corners[2] = cvPoint( src.cols, src.rows ); obj_corners[3] = cvPoint( 0, src.rows );
			std::vector<Point2f> scene_corners(4);
			perspectiveTransform( obj_corners, scene_corners, homo);
			for (size_t i = 0; i < homo.rows; ++i)
			{
				for (size_t j = 0; j < homo.cols; ++j)
					printf("%lf ", homo.at<double>(i, j));
				printf("\n");
			}
			line(frameImg,scene_corners[0],scene_corners[1],CV_RGB(255,0,0),2);  
			line(frameImg,scene_corners[1],scene_corners[2],CV_RGB(255,0,0),2);  
			line(frameImg,scene_corners[2],scene_corners[3],CV_RGB(255,0,0),2);  
			line(frameImg,scene_corners[3],scene_corners[0],CV_RGB(255,0,0),2);  
			return true;  
		}
		return true;
	}  


}  
int main()  
{  
	Mat orig;
	string filename = "image20.jpg";  
	src = imread(filename,0);  
	orig = imread(filename);

	width = src.cols;  
	height = src.rows;  
	string detectorType = "SIFT";  
	string descriptorType = "SIFT";  
	string matcherType = "FlannBased";  

	Ptr<FeatureDetector> featureDetector;  
	Ptr<DescriptorExtractor> descriptorExtractor;  
	Ptr<DescriptorMatcher> descriptorMatcher;  
	if( !createDetectorDescriptorMatcher( detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptorMatcher ) )  
	{  
		cout<<"Creat Detector Descriptor Matcher False!"<<endl;  
		return -1;  
	}  
	//Intial: read the pattern img keyPoint  
	vector<KeyPoint> queryKeypoints;  
	Mat queryDescriptor;  
	featureDetector->detect(src,queryKeypoints);  
	descriptorExtractor->compute(src,queryKeypoints,queryDescriptor);  

	//VideoCapture cap(0); // open the default camera  
	//cap.set( CV_CAP_PROP_FRAME_WIDTH,320);
	//cap.set( CV_CAP_PROP_FRAME_HEIGHT,240 );
	//if(!cap.isOpened())  // check if we succeeded  
	//{  
	//	cout<<"Can't Open Camera!"<<endl;  
	//	return -1;  
	//}  
	srcCorner[0] = Point(0,0);  
	srcCorner[1] = Point(width,0);  
	srcCorner[2] = Point(width,height);  
	srcCorner[3] = Point(0,height);  

	vector<KeyPoint> trainKeypoints;  
	Mat trainDescriptor;  

	Mat frame,grayFrame;
	Mat warpedFrame;
	Mat homo;
	char key=0;  

	frame = imread("image21.jpg");  
	while (key!=27)  
	{  
		//cap>>frame;  
		if (!frame.empty())
		{
			frame.copyTo(frameImg);
			printf("%d,%d\n",frame.depth(),frame.channels());
			grayFrame.zeros(frame.rows,frame.cols,CV_8UC1);
			cvtColor(frame,grayFrame,CV_BGR2GRAY);  
			trainKeypoints.clear();  
			trainDescriptor.setTo(0);  
			featureDetector->detect(grayFrame,trainKeypoints);  

			if(trainKeypoints.size()!=0)  
			{  
				descriptorExtractor->compute(grayFrame,trainKeypoints,trainDescriptor);  

				bool isFound = matchingDescriptor(queryKeypoints,trainKeypoints,queryDescriptor,trainDescriptor,descriptorMatcher, homo);  
				imshow("foundImg",frameImg);
				imshow("src", orig);

				
				warpPerspective(orig, warpedFrame, homo, cvSize(1000, 1000));
				imshow("Merged", warpedFrame);
				
				Mat image1s, image2s;
				frameImg.convertTo(image1s, CV_16S);
				warpedFrame.convertTo(image2s, CV_16S);

				Mat mask1(image1s.size(), CV_8U);
				//mask1(Rect(0, 0, mask1.cols/2, mask1.rows)).setTo(255);
				//mask1(Rect(mask1.cols/2, 0, mask1.cols - mask1.cols/2, mask1.rows)).setTo(0);
				Mat mask2(image2s.size(), CV_8U);
				//mask2(Rect(0, 0, mask2.cols/2, mask2.rows)).setTo(0);
				//mask2(Rect(mask2.cols/2, 0, mask2.cols - mask2.cols/2, mask2.rows)).setTo(255);
				detail::MultiBandBlender blender(false, 5);

				blender.prepare(Rect(0, 0, max(image1s.cols, image2s.cols), max(image1s.rows, image2s.rows)));
				blender.feed(image1s, mask1, Point(0,0));
				blender.feed(image2s, mask2, Point(0,0));

				Mat result_s, result_mask;
				blender.blend(result_s, result_mask);
				Mat result; result_s.convertTo(result, CV_8U);

				imshow("Merged", result);

			}  
		}
		
		
		key = waitKey(100);
		if (key == 32) {
			waitKey(0);
		}
	}  
	//cap.release();
	return 0;
}  
//int main()    
//{  
//    vector< Mat > vImg;  
//    Mat rImg;  
//    namedWindow("Stitching Result");
//	
//	
//    vImg.push_back( imread("image20.jpg") );  
//    vImg.push_back( imread("image21.jpg") );
//    //vImg.push_back( imread("./stitching_img/S3.jpg") );  
//    //vImg.push_back( imread("./stitching_img/S4.jpg") );  
//    //vImg.push_back( imread("./stitching_img/S5.jpg") );  
//    //vImg.push_back( imread("./stitching_img/S6.jpg") );  
//        
//      
//    Stitcher stitcher = Stitcher::createDefault();  
//      
//      
//    unsigned long AAtime=0, BBtime=0; //check processing time  
//    AAtime = getTickCount(); //check processing time  
//      
//    Stitcher::Status status = stitcher.stitch(vImg, rImg);  
//
//    BBtime = getTickCount(); //check processing time   
//    printf("%.2lf sec \n",  (BBtime - AAtime)/getTickFrequency() ); //check processing time  
//	imshow("Sti0", vImg[0]); 
//    imshow("Sti1", vImg[1]);
//	waitKey(0);
//
//	if (status != Stitcher::OK)
//    {
//        cout << "Can't stitch images, error code = " << status << endl;
//		waitKey(0);
//        return -1;
//    }
//    imshow("Stitching Result", rImg);  
//      
//    waitKey(0);  
//      
//}    