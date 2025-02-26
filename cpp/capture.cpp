#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // 创建会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 加载 ONNX 模型
    const char* model_path = "2dfan4.onnx";  // 替换为你的模型路径
    Ort::Session session(env, model_path, session_options);

    std::cout << "ONNX Runtime initialized successfully!" << std::endl;
    return 0;
}