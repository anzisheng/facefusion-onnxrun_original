#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp> // For image processing
#include <vector>
#include <cmath>
#include <string>

enum FaceShape {
    OVAL,
    ROUND,
    SQUARE,
    HEART,
    LONG,
    DIAMOND,
    UNKNOWN
};

struct FaceLandmarks {
    std::vector<float> landmarks; // 68 points x,y coordinates (136 values total)
    cv::Rect face_rect;
};

// Helper function to run ONNX model inference
FaceLandmarks detect_landmarks_onnx(Ort::Session& session, const cv::Mat& image) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FaceShape");
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    
    // Preprocess image (adjust based on your model's requirements)
    cv::Mat resized, normalized;
    cv::resize(image, resized, cv::Size(256, 256)); // Typical input size for 2DFAN4
    resized.convertTo(normalized, CV_32FC3, 1.0/255.0);
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, 256, 256};
    std::vector<float> input_data(normalized.total() * normalized.channels());
    cv::Mat flat = normalized.reshape(1, normalized.total() * normalized.channels());
    input_data.assign(flat.begin<float>(), flat.end<float>());
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), 
        input_shape.data(), input_shape.size());
    
    // Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"landmarks_xyscore"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},  
                                   input_names, &input_tensor, 1, 
                                   output_names, 1);
    
    // Process output (adjust based on your model's output format)
    float* output = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    FaceLandmarks result;
    result.landmarks.assign(output, output + output_size);
    
    // Calculate approximate face rectangle from landmarks
    float min_x = FLT_MAX, min_y = FLT_MAX, max_x = 0, max_y = 0;
    for (size_t i = 0; i < 68; ++i) {
        float x = result.landmarks[i*2];
        float y = result.landmarks[i*2+1];
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
    }
    result.face_rect = cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y);
    
    return result;
}

FaceShape classify_face_shape(const FaceLandmarks& landmarks) {
    // Extract key points (indices same as dlib 68-point model)
    auto get_point = [&](int idx) -> cv::Point2f {
        return cv::Point2f(landmarks.landmarks[idx*2], landmarks.landmarks[idx*2+1]);
    };
    
    // Calculate measurements
    float jaw_width = get_point(16).x - get_point(0).x;
    float face_height = get_point(8).y - get_point(27).y;
    float cheekbone_width = get_point(14).x - get_point(2).x;
    float forehead_width = get_point(21).x - get_point(17).x;
    
    // Calculate ratios
    float jaw_to_height = jaw_width / face_height;
    float cheek_to_jaw = cheekbone_width / jaw_width;
    float forehead_to_jaw = forehead_width / jaw_width;
    
    // Calculate jaw angle
    float left_jaw_angle = std::atan2(get_point(3).y - get_point(4).y, 
                                    get_point(3).x - get_point(4).x);
    float right_jaw_angle = std::atan2(get_point(13).y - get_point(12).y, 
                                     get_point(13).x - get_point(12).x);
    float jaw_angle = std::abs(left_jaw_angle - right_jaw_angle);
    
    // Classification rules (adjust thresholds as needed)
    if (jaw_to_height > 0.75f && jaw_angle > 1.0f) {
        return SQUARE;
    }
    else if (jaw_to_height < 0.65f && forehead_to_jaw > 1.1f && cheek_to_jaw > 1.05f) {
        return HEART;
    }
    else if (jaw_to_height < 0.7f && cheek_to_jaw > 1.0f && jaw_angle < 0.8f) {
        return OVAL;
    }
    else if (std::abs(jaw_width - cheekbone_width) < 0.1f*jaw_width && jaw_to_height > 0.7f) {
        return ROUND;
    }
    else if (face_height > 1.4f * cheekbone_width) {
        return LONG;
    }
    else if (cheekbone_width > jaw_width && forehead_width > jaw_width && 
             cheekbone_width > forehead_width) {
        return DIAMOND;
    }
    
    return UNKNOWN;
}

std::string face_shape_to_string(FaceShape shape) {
    switch(shape) {
        case OVAL: return "Oval";
        case ROUND: return "Round";
        case SQUARE: return "Square";
        case HEART: return "Heart";
        case LONG: return "Long";
        case DIAMOND: return "Diamond";
        default: return "Unknown";
    }
}

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FaceShape");
    Ort::SessionOptions session_options;
     std::cout <<"hello 000111" <<std::endl;
    // Load the ONNX model
    Ort::Session session(env, "2dfan4.onnx", session_options);
    
    // Load input image (replace with your image loading)
    cv::Mat image = cv::imread("face.jpg");
    std::cout <<"hello 000" <<std::endl;
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }
     std::cout <<"hello 000333" <<std::endl;
    
    // Detect landmarks
    FaceLandmarks landmarks = detect_landmarks_onnx(session, image);
    
    // Classify face shape
    FaceShape shape = classify_face_shape(landmarks);
    std::cout << "Detected face shape: " << face_shape_to_string(shape) << std::endl;
    
    return 0;
}