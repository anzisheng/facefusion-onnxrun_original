#include "yolov8face.h"
#include "face68landmarks.h"
#include "facerecognizer.h"
#include "faceswap.h"
#include "faceenhancer.h"
#include "Stopwatch.h"
using namespace cv;
using namespace std;
#include <filesystem>
namespace fs = std::filesystem;
#include <fmt/core.h>

//#include "buffers.h"
//#include "faceswap_fromMNist.h"
//#include "cmd_line_parser.h"
//#include "logger.h"
//#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include "yolov8face.h"
//#include "faceswap.h"
#include "faceswap.h"
//#include "face68landmarks.h"
#include "Face68Landmarks.h"
#include "facerecognizer.h"
#include "faceenhancer.h"
//#include "faceenhancer.h"
//#include "faceenhancer_trt.h"
//#include "faceenhancer_trt2.h"
#include "faceswap.h"
//#include "SampleOnnxMNIST.h"
#include "nlohmann/json.hpp" 

//#include "faceswap.h"
//#include "engine.h"
//#include "utile.h"
#include "yolov8face.h"
#include <vector>

#include "nlohmann/json.hpp"
#include<string>
#include<iostream>
#include<fstream>
#include <json/json.h>
#include <iostream>
#include <json/json.h>
using namespace std;
using namespace Json;
using namespace std;
using json =nlohmann::json;

#include <websocketpp/config/asio_no_tls.hpp>

#include <websocketpp/server.hpp>

#include <iostream>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

// pull out the type of messages sent by our config
typedef server::message_ptr message_ptr;


///////////////////////
class TaskSocket
{
    public:
    string photo;
    string style;
    TaskSocket(string photo_file, string style_file):photo(photo_file), style(style_file){}

};
std::queue<TaskSocket> messageQueue; // 消息队列
std::mutex mtx; // 互斥锁
std::condition_variable cvs; // 条件变量
//////////////////////////
class TaskResult
{
    public:
    string result_name;
    //string style;
    TaskResult(string result):result_name(result){};

};
std::queue<TaskResult> resultQueue; // 消息队列
std::mutex mtx_result; // 互斥锁
std::condition_variable cvs_result; // 条件变量

////////////////////////////
Yolov8Face* detect_face_net = NULL;
Face68Landmarks* detect_68landmarks_net = NULL;
FaceEmbdding* face_embedding_net = NULL;
SwapFace*  swap_face_net = NULL;
FaceEnhance* enhance_face_net = NULL;

void init_model()
{
    //FaceEnhance enhance_face_net = NULL;
	////图片路径和onnx文件的路径，要确保写正确，才能使程序正常运行的
	if(detect_face_net == NULL)
		detect_face_net = new Yolov8Face("weights/yoloface_8n.onnx");
	////cout << "www11hello000"<<endl;
	if(detect_68landmarks_net == NULL)
		detect_68landmarks_net = new Face68Landmarks("weights/2dfan4.onnx");
	////cout << "1111"<<endl;
	if(face_embedding_net == NULL)
		face_embedding_net = new FaceEmbdding("weights/arcface_w600k_r50.onnx");
	////cout << "2222"<<endl;
	if(swap_face_net == NULL)
		swap_face_net = new SwapFace ("weights/inswapper_128.onnx");
	if(enhance_face_net == NULL)
		enhance_face_net = new FaceEnhance ("weights/gfpgan_1.4.onnx");
}

void free_faces()
{
    //Yolov8Face* detect_face_net = NULL;
	if(detect_face_net) delete detect_face_net;
	//Face68Landmarks* detect_68landmarks_net = NULL;
	if(detect_68landmarks_net) delete detect_68landmarks_net;
	//FaceEmbdding* face_embedding_net = NULL;
	if(face_embedding_net) delete face_embedding_net;
	//SwapFace*  swap_face_net = NULL;
	if(swap_face_net) delete swap_face_net;
	//FaceEnhance* enhance_face_net = NULL;
	if(enhance_face_net) delete enhance_face_net;
    //if(yoloV8) delete yoloV8;
}
string swap_faces(string photo, string style){
	////cout << "hello000"<<endl;
	string source_path = photo;
	string target_path = style;
	
	
	
	////cout << "wwww999"<<endl;
	preciseStopwatch stopwatch;
	Mat source_img = imread(source_path);
	Mat target_img = imread(target_path);

    vector<Bbox> boxes;

	detect_face_net->detect(source_img, boxes);
	////cout << "hello12244"<<endl;
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> face_landmark_5of68;
	
	vector<Point2f> face68landmarks = detect_68landmarks_net->detect(source_img, boxes[position], face_landmark_5of68);
	
	vector<float> source_face_embedding = face_embedding_net->detect(source_img, face_landmark_5of68);
	
	detect_face_net->detect(target_img, boxes);
	position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> target_landmark_5;
	detect_68landmarks_net->detect(target_img, boxes[position], target_landmark_5);
	

	Mat swapimg = swap_face_net->process(target_img, source_face_embedding, target_landmark_5);
	
	Mat resultimg = enhance_face_net->process(swapimg, target_landmark_5);
	//string result = fmt::format("{}_{}.jpg",  photo.substr(0, photo.rfind(".")), style.substr(0, style.rfind(".")));

    //string result = photo.substr(0, photo.rfind("."))+"_"+style;//fmt::format("{}_{}.jpg",  photo.substr(0, photo.rfind(".")), style.substr(0, style.rfind(".")));
    
    fs::current_path("./");
    fs::path currentPath = fs::current_path();
	std::cout << currentPath << std::endl;
    std::cout << "currentPath:" << currentPath.string() << std::endl;

    
    string file = photo;
    int pos = file.find_last_of('/');
    cout << "pos of photo is " << pos <<endl;
    std::string path_photo(file.substr(0, pos));
    std::string name_photo(file.substr(pos + 1));
    name_photo = name_photo.substr(0, name_photo.rfind("."));
    cout << "name photp: " << name_photo<<endl;
    
    file = style;
    pos = file.find_last_of('/');
    cout << "pos of style is " << pos <<endl;
    std::string path_style((pos < 0)? "" : file.substr(0, pos));
    std::string name_style(file.substr(pos + 1));
    //name_photo = name_style.substr(0, name_style.rfind("."));
    cout << "name style: " << name_style<<endl;
           

    std::cout << "file photo path is: " << path_photo << std::endl;
    std::cout << "file style path is: " << path_style << std::endl;
    std::string temp = name_photo.substr(0, name_photo.rfind(".")) +"_"+name_style.substr(0, name_style.rfind("."))+".jpg";
    std::cout << "new jpg name := " << temp << std::endl;

    std::filesystem::path temp_fs_path_append(path_photo+"/"+path_style+"/"+temp);

    
    //string result = name_photo.substr(0, name_photo.rfind(".")) +"_"+name_style.substr(0, name_style.rfind("."))+".jpg";
    //std::cout << "at last jpg name" << result << std::endl;
    //temp_fs_path_append.append(result);
    std::cout << "combined path :" << temp_fs_path_append << std::endl;
    currentPath.append(temp_fs_path_append.string());
    std::cout << "currentPath:" << currentPath.string() << std::endl;
    //std::filesystem::path p(temp);
    //cout << "path p: " <<p.string() <<endl;    
    std::filesystem::create_directories(currentPath.parent_path());
    cout << "path p's parent: " <<currentPath.parent_path() <<endl;    
    

    //cout << "result name: " <<result <<endl;
    //std::filesystem::path outputPath = p;//+result;
    //cout << "last name: " <<outputPath <<endl;
    //std::ofstream outputFile(outputPath, std::ios_base::app); 
    //string result = temp+name_style;
    //cout << "result: " <<temp_fs_path_append.string() <<endl;
    imwrite(currentPath.string(), resultimg);
	// auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    // cout << "total time is " << totalElapsedTimeMs/1000 <<" S"<<endl;

	    

    return currentPath.string();//result;


}

// 生产者线程函数，向消息队列中添加消息
void producerFunction(Json::Value &root) {
    cout << "hello: " << root <<std::endl;
    int StyleNum = root["styleName"].size();
    string PhotoName = root["sessionID"].asString()+"/0.jpg";
    for (int i = 0; i < StyleNum; i++)
    {
        /* code */
        //TaskSocket message(PhotoName, root["styleName"][i]["name"].asString());
        swap_faces( PhotoName, root["styleName"][i]["name"].asString());

        //将消息添加到队列
        {
            //std::lock_guard<std::mutex> lock(mtx);
            //messageQueue.push(message);
            //std::cout << "Produced message: " << message.photo<<"," <<message.style << std::endl;
        }
           // 通知等待的消费者线程
         cvs.notify_one();

        // 模拟一些工作
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
     }

    }

// 消费者线程函数，从消息队列中获取消息
void consumerFunction(server* s, websocketpp::connection_hdl hdl,message_ptr msg) {
    preciseStopwatch stopwatch;
    while (true) {
        // 等待消息队列非空
        cout << "queue size:" << messageQueue.size() <<std::endl;
        std::unique_lock<std::mutex> lock(mtx);
        cvs.wait(lock, [] { return !messageQueue.empty(); });

        // 从队列中获取消息
        TaskSocket message = messageQueue.front();
        messageQueue.pop();
        std::cout << "Consumed message: " << message.photo <<" and " <<message.style << std::endl;
//        cout << "begin swap_faces(message.photo,message.style)" ;
  
        // 检查是否为终止信号
        //  if (message.style == "-10.jpg") {
        //      break;
        //  }
        string swap_result = swap_faces(message.photo,message.style);
        
        cout << "swap_result:   " << swap_result <<endl;
        TaskResult  reultMsg = TaskResult(swap_result);        
        //将消息添加到队列
        //{
            //std::lock_guard<std::mutex> lock(mtx_result);
            //resultQueue.push(reultMsg);
            //std::cout << "swap_faces Produced result: " << reultMsg.result_name << std::endl;
        //}
           // 通知等待的消费者线程
         //cvs_result.notify_one();

        
        // 检查是否为终止信号
        //  if (message.style == "-11.jpg") {
        //      break;
        //  }
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
        //Json::Value root; 
        //TaskResult message = resultQueue.front();
        //resultQueue.pop();
        // 向对象中添加数据
        // root["type"] = "Generating>>>>!";
        // root["result_name"] = swap_result;//message.result_name; 
        // // 创建一个Json::StreamWriterBuilder
        // Json::StreamWriterBuilder writer;
        // // 将Json::Value对象转换为字符串
        // std::string output = Json::writeString(writer, root);
    
        // // 打印输出
        // //std::cout << output << std::endl;
        // //s->send(hdl, msg->get_payload(), msg->get_opcode());
        // s->send(hdl, output, msg->get_opcode());
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    //free the space.
    
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++=================================="<<endl;
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    cout << "total time is " << totalElapsedTimeMs/1000 <<" S"<<endl;
    //free_faces();
}






// Define a callback to handle incoming messages
void on_message(server* s, websocketpp::connection_hdl hdl, message_ptr msg) {
    std::cout << "on_message called with hdl: " << hdl.lock().get()
              << " and message: " << msg->get_payload()
              << std::endl;
    nlohmann::json commands = msg->get_payload().data();
    //std::cout << "to raw string:"<<commands << std::endl;
    std::string jsonString = commands;
    // 创建一个Json::CharReaderBuilder
    Json::CharReaderBuilder builder;

    // 创建一个Json::Value对象
    Json::Value root;
 
    // 创建一个错误信息字符串
    std::string errors;
    
    //Json::Value val;
    //Json::Reader reader;
    
    // 解析JSON字符串
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parsingSuccessful = reader->parse(jsonString.c_str(), jsonString.c_str() + jsonString.size(), &root, &errors);
    if (!parsingSuccessful) {
            // 打印错误信息并退出
            std::cout << "Error parsing JSON: " << errors << std::endl;
            //return 1;
        }
    
    Json::Value root2; 
    //producerFunction(root);
    int StyleNum = root["styleName"].size();
    string PhotoName = root["sessionID"].asString()+"/0.jpg";
    for (int i = 0; i < StyleNum; i++)
    {
        /* code */
        //TaskSocket message(PhotoName, root["styleName"][i]["name"].asString());
        swap_faces( PhotoName, root["styleName"][i]["name"].asString());

        //将消息添加到队列
        {
            //std::lock_guard<std::mutex> lock(mtx);
            //messageQueue.push(message);
            //std::cout << "Produced message: " << message.photo<<"," <<message.style << std::endl;
        }
           // 通知等待的消费者线程
         //cvs.notify_one();

        // 模拟一些工作
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
         root2["type"] = "Generating!";
        root2["result_name"] = "abc.jpg";//currentPath.string();//result;//message.result_name; 
        // 创建一个Json::StreamWriterBuilder
         Json::StreamWriterBuilder writer;
        // // 将Json::Value对象转换为字符串
         std::string output = Json::writeString(writer, root2);
        
        // 打印输出
        //std::cout << output << std::endl;
        //s->send(hdl, msg->get_payload(), msg->get_opcode());
            
     }
     root2["type"] = "Generating!";
     root2["result_name"] = "abcd.jpg";//currentPath.string();//result;//message.result_name; 
     // 创建一个Json::StreamWriterBuilder
     Json::StreamWriterBuilder writer;
     // 将Json::Value对象转换为字符串
     std::string output = Json::writeString(writer, root2);
     
     // 打印输出
     //std::cout << output << std::endl;
     //s->send(hdl, msg->get_payload(), msg->get_opcode());
     s->send(hdl, output, msg->get_opcode());


    
    //TaskResult message = resultQueue.front();
    //resultQueue.pop();
    // 向对象中添加数据
    
    //std::this_thread::sleep_for(std::chrono::milliseconds(1));
 
    //std::thread producer(producerFunction, std::ref(root));
    //std::thread consumer(consumerFunction, s, hdl,  msg);

    // 等待线程执行完成
    //consumer.join();
    //producer.join();
    
    cout << "-----------------------"<<endl;

    //json commands = json(msg->get_payload().data())["sessionID"];
    //std::cout << "the sessioId is :"<<commands.at("sessionID")<<std::endl;

    //order cr;
    //from_json(commands, cr);
    Json::Value root3;
    root3["type"] = "Complete!";
    //root["result_name"] = message.result_name; 
    Json::StreamWriterBuilder writer2;
    // 将Json::Value对象转换为字符串
    std::string output2 = Json::writeString(writer2, root3);
    s->send(hdl, output2, msg->get_opcode()); 

    std::cout <<"waiting.... for post next order!"<<std::endl;

    // check for a special command to instruct the server to stop listening so
    // it can be cleanly exited.
    if (msg->get_payload() == "stop-listening") {
        s->stop_listening();
        return;
    }
    /*
    while(!resultQueue.empty())
    {

    try {
    // std::cout << "reuren message" <<std::endl;
    // nlohmann::json commands = msg->get_payload().data();
    // std::cout << "to raw string:"<<commands << std::endl;
    // std::string jsonString = commands;
    // // 创建一个Json::CharReaderBuilder
    //Json::CharReaderBuilder builder;

    // 创建一个Json::Value对象
    // Json::Value root;
 
    // // 创建一个错误信息字符串
    // std::string errors;
    
    // //Json::Value val;
    // //Json::Reader reader;
    
    // // 解析JSON字符串
    // std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    // bool parsingSuccessful = reader->parse(jsonString.c_str(), jsonString.c_str() + jsonString.size(), &root, &errors);
    // if (!parsingSuccessful) {
    //         // 打印错误信息并退出
    //         std::cout << "Error parsing JSON: " << errors << std::endl;
    //         //return 1;
    //     }
    
    // // 提取并打印数据
    // std::cout << "Name: " << root["sessionID"].asString() << std::endl;
    // int numStyle = root["styleName"].size();
    // std::cout << "styleName size: " << numStyle << std::endl;
    // for(int i = 0; i < numStyle; i++)
    // {
    //     std::cout << root["styleName"][i]["name"].asString()<<std::endl;
    // }
    // 创建一个Json::Value对象
    
   std::cout << "resultQueue size :" <<resultQueue.size() << std::endl;
    {
        std::unique_lock<std::mutex> lock(mtx_result);
        cvs_result.wait(lock, [] { return !resultQueue.empty(); });
    }
    Json::Value root; 
    TaskResult message = resultQueue.front();
    resultQueue.pop();
    // 向对象中添加数据
    root["type"] = "Generating.....!";
    root["result_name"] = message.result_name; 
    // 创建一个Json::StreamWriterBuilder
    Json::StreamWriterBuilder writer;
    // 将Json::Value对象转换为字符串
    std::string output = Json::writeString(writer, root);
 
    // 打印输出
    //std::cout << output << std::endl;
    //s->send(hdl, msg->get_payload(), msg->get_opcode());
        s->send(hdl, output, msg->get_opcode());
    } catch (websocketpp::exception const & e) {
        std::cout << "Echo failed because: "
                  << "(" << e.what() << ")" << std::endl;
    }
    }*/
       
 

}



#include <openssl/x509.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <iostream>
#include <ctime>

// 检查证书是否过期
bool is_certificate_expired(X509* cert) {
    if (!cert) {
        return true; // 无效证书
    }

    // 获取证书的过期时间
    ASN1_TIME* expiry_time = X509_get_notAfter(cert);
    if (!expiry_time) {
        return true; // 无法获取过期时间
    }

    // 将 ASN1_TIME 转换为 time_t
    struct tm tm_expiry;
    if (!ASN1_TIME_to_tm(expiry_time, &tm_expiry)) {
        return true; // 转换失败
    }

    // 获取当前时间
    std::time_t now = std::time(nullptr);
    std::tm tm_now = *std::gmtime(&now);

    // 比较时间
    if (std::mktime(&tm_expiry) < std::mktime(&tm_now)) {
        return true; // 证书已过期
    }

    return false; // 证书未过期
}

int main() {

//   std::string file = "abc/images/1.jpg";
//   int pos = file.find_last_of('/');
//   std::string path(file.substr(0, pos));
//   std::string name(file.substr(pos + 1));
//   std::cout << "file path is: " << path << std::endl;
//   std::cout << "file name is: " << name << std::endl;
    
    const char* cert_file = "server.crt";

    // 初始化 OpenSSL
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();
    // 打开证书文件
    FILE* fp = fopen(cert_file, "r");
    if (!fp) {
        std::cerr << "无法打开证书文件: " << cert_file << std::endl;
        return 1;
    }
   // 加载证书
    X509* cert = PEM_read_X509(fp, nullptr, nullptr, nullptr);
    fclose(fp);

    if (!cert) {
        std::cerr << "无法加载证书" << std::endl;
        ERR_print_errors_fp(stderr);
        return 1;
    }

    // 检查证书是否过期
    if (is_certificate_expired(cert)) {
        std::cout << "证书已过期。请更新证书！ " << std::endl;
        X509_free(cert);
        EVP_cleanup();
        ERR_free_strings();
        return 0;
    } else {
        std::cout << "证书未过期。请提交订单!" << std::endl;
    }

    // 释放资源
    X509_free(cert);
    EVP_cleanup();
    ERR_free_strings();
	//return 0;


	//////////////////////////
	// Create a server endpoint
    server echo_server;
    init_model();
    std::cout << "hello, I am a cpu server!"<< std::endl;
	try {
        // Set logging settings
        echo_server.set_access_channels(websocketpp::log::alevel::all);
        echo_server.clear_access_channels(websocketpp::log::alevel::frame_payload);

        // Initialize Asio
        echo_server.init_asio();

        // Register our message handler
        echo_server.set_message_handler(bind(&on_message,&echo_server,::_1,::_2));

        // Listen on port 9002
        echo_server.listen(9002);

        // Start the server accept loop
        echo_server.start_accept();

        // Start the ASIO io_service run loop
        echo_server.run();
        cout << "handling over, free the model."<<endl;
        free_faces();
    } catch (websocketpp::exception const & e) {
        std::cout << e.what() << std::endl;
    } catch (...) {
        std::cout << "other exception" << std::endl;
    }

}