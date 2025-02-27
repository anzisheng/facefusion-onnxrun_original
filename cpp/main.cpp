#include "yolov8face.h"
#include "face68landmarks.h"
#include "facerecognizer.h"
#include "faceswap.h"
#include "faceenhancer.h"
#include "Stopwatch.h"
using namespace cv;
using namespace std;

int main()
{
	////cout << "hello000"<<endl;
	string source_path = "images/5.jpg";
	string target_path = "images/target.jpg";
    ////cout << "wwww000"<<endl;
	
	////图片路径和onnx文件的路径，要确保写正确，才能使程序正常运行的
	Yolov8Face detect_face_net("weights/yoloface_8n.onnx");
	////cout << "www11hello000"<<endl;
	Face68Landmarks detect_68landmarks_net("weights/2dfan4.onnx");
	////cout << "1111"<<endl;
	FaceEmbdding face_embedding_net("weights/arcface_w600k_r50.onnx");
	////cout << "2222"<<endl;
	SwapFace swap_face_net("weights/inswapper_128.onnx");
	FaceEnhance enhance_face_net("weights/gfpgan_1.4.onnx");
	////cout << "wwww999"<<endl;
	preciseStopwatch stopwatch;
	Mat source_img = imread(source_path);
	Mat target_img = imread(target_path);
	//preciseStopwatch stopwatch;
	////cout << "hello111: " <<source_img.rows<<source_img.cols <<endl;

    vector<Bbox> boxes;
	////cout << "hello12222"<<endl;
	detect_face_net.detect(source_img, boxes);
	////cout << "hello12244"<<endl;
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> face_landmark_5of68;
	////cout << "hello12255"<<endl;
	vector<Point2f> face68landmarks = detect_68landmarks_net.detect(source_img, boxes[position], face_landmark_5of68);
	////cout << "hello133"<<endl;
	vector<float> source_face_embedding = face_embedding_net.detect(source_img, face_landmark_5of68);
	////cout << "hello222"<<endl;
	detect_face_net.detect(target_img, boxes);
	position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> target_landmark_5;
	detect_68landmarks_net.detect(target_img, boxes[position], target_landmark_5);
	////cout << "hello7777"<<endl;

	Mat swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5);
	////cout << "hello888"<<endl;
	Mat resultimg = enhance_face_net.process(swapimg, target_landmark_5);
	////cout << "hello999999"<<endl;
	imwrite("resultimg.jpg", resultimg);
	auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    cout << "total time is " << totalElapsedTimeMs/1000 <<" S"<<endl;

	/*static const string kWinName = "Deep learning face swap use onnxruntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, resultimg);
	waitKey(0);
	destroyAllWindows();*/
}