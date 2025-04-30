#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
using namespace cv;
using namespace dlib;

int main() {
    // 加载Dlib模型
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor predictor;
    deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

    Mat image = imread("input.jpg");
    cv_image<bgr_pixel> dlib_img(image);

    // 检测人脸
    std::vector<rectangle> faces = detector(dlib_img);
    for (const auto& face : faces) {
        // 获取68个关键点
        full_object_detection landmarks = predictor(dlib_img, face);

        // 提取左眼和右眼区域（点36-41和42-47）
        std::vector<Point> left_eye, right_eye;
        for (int i = 36; i <= 41; i++) 
            left_eye.push_back(Point(landmarks.part(i).x(), landmarks.part(i).y()));
        for (int i = 42; i <= 47; i++) 
            right_eye.push_back(Point(landmarks.part(i).x(), landmarks.part(i).y()));

        // 计算眼睛ROI的边界框
        Rect left_eye_roi = boundingRect(left_eye);
        Rect right_eye_roi = boundingRect(right_eye);

        // 显示结果
        rectangle(image, left_eye_roi, Scalar(0, 255, 0), 2);
        rectangle(image, right_eye_roi, Scalar(0, 255, 0), 2);
    }

    imshow("Eyes Detection", image);
    waitKey(0);
    return 0;
}