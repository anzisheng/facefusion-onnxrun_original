#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <iostream>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <json/json.h>
using namespace std;
using namespace Json;

typedef websocketpp::server<websocketpp::config::asio> server;
// pull out the type of messages sent by our config
typedef server::message_ptr message_ptr;

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

// 全局变量，用于存储连接句柄
std::queue<websocketpp::connection_hdl> connection_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;

// 定义回调函数，处理新连接
void on_open(server* s, websocketpp::connection_hdl hdl) {
    std::cout << "New connection established!" << std::endl;

    // 将连接句柄存入队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        connection_queue.push(hdl);
        queue_cv.notify_one(); // 通知发送线程
    }
}

// 发送消息的线程函数
void send_messages(server* s) {
    while (true) {
        websocketpp::connection_hdl hdl;

        // 从队列中获取连接句柄
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [] { return !connection_queue.empty(); });
            hdl = connection_queue.front();
            connection_queue.pop();
        }

        // 构造消息
        std::string message = "Hello from server! This is a direct message.";
        Json::Value root2; 
        root2["type"] = "Generating!";
        root2["result_name"] = "abc.jpg";//currentPath.string();//result;//message.result_name; 
        // 创建一个Json::StreamWriterBuilder
        Json::StreamWriterBuilder writer;
        // // 将Json::Value对象转换为字符串
        std::string output = Json::writeString(writer, root2);
        //s->send(hdl, output, msg->get_opcode());        

        // 使用 post 方法将发送操作放到 ASIO 事件循环中
        s->get_io_service().post([s, hdl, output]() {
            try {
                s->send(hdl, output, websocketpp::frame::opcode::text);
            } catch (websocketpp::exception const & e) {
                std::cout << "Send failed: " << e.what() << std::endl;
            }
        });

        // 模拟延迟
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
// Define a callback to handle incoming messages
void on_message(server* s, websocketpp::connection_hdl hdl, message_ptr msg) {
    std::cout << "on_message called with hdl: " << hdl.lock().get()
              << " and message: " << msg->get_payload()
              << std::endl;

   /*           
            Json::Value root2; 
            root2["type"] = "Generating!";
            root2["result_name"] = "abc.jpg";//currentPath.string();//result;//message.result_name; 
            // 创建一个Json::StreamWriterBuilder
            Json::StreamWriterBuilder writer;
            // // 将Json::Value对象转换为字符串
            std::string output = Json::writeString(writer, root2);
            //s->send(hdl, output, msg->get_opcode());        
    
            // 使用 post 方法将发送操作放到 ASIO 事件循环中
            s->get_io_service().post([s, hdl, output]() {
                try {
                    s->send(hdl, output, websocketpp::frame::opcode::text);
                } catch (websocketpp::exception const & e) {
                    std::cout << "Send failed: " << e.what() << std::endl;
                }
            });


            std::this_thread::sleep_for(std::chrono::milliseconds(10000));

            //Json::Value root2; 
            root2["type"] = "Generating00000!";
            root2["result_name"] = "abc.jpg";//currentPath.string();//result;//message.result_name; 
            // 创建一个Json::StreamWriterBuilder
            //Json::StreamWriterBuilder writer;
            // // 将Json::Value对象转换为字符串
            //std::string 
            output = Json::writeString(writer, root2);
            
            //s->get_io_service().post([s, hdl, output]() 
            {
                try {
                    s->send(hdl, output, websocketpp::frame::opcode::text);
                } catch (websocketpp::exception const & e) {
                    std::cout << "Send failed: " << e.what() << std::endl;
                }
            }
        //);
            std::this_thread::sleep_for(std::chrono::milliseconds(15000));
            root2["type"] = "Generating1111!";
            root2["result_name"] = "abc.jpg";//currentPath.string();//result;//message.result_name; 
            // 创建一个Json::StreamWriterBuilder
            //Json::StreamWriterBuilder writer;
            // // 将Json::Value对象转换为字符串
            //std::string 
            output = Json::writeString(writer, root2);
            
            //s->get_io_service().post([s, hdl, output]() 
            {
                try {
                    s->send(hdl, output, websocketpp::frame::opcode::text);
                } catch (websocketpp::exception const & e) {
                    std::cout << "Send failed: " << e.what() << std::endl;
                }
            }
        //);
            std::this_thread::sleep_for(std::chrono::milliseconds(15000));
            root2["type"] = "Generating2222!";
            root2["result_name"] = "abc.jpg";//currentPath.string();//result;//message.result_name; 
            // 创建一个Json::StreamWriterBuilder
            //Json::StreamWriterBuilder writer;
            // // 将Json::Value对象转换为字符串
            //std::string 
            output = Json::writeString(writer, root2);
            
            //s->get_io_service().post([s, hdl, output]() 
            {
                try {
                    s->send(hdl, output, websocketpp::frame::opcode::text);
                } catch (websocketpp::exception const & e) {
                    std::cout << "Send failed: " << e.what() << std::endl;
                }
            }
        //);
        */
       std::thread([s, hdl]() {
        // 发送第一条消息
        s->send(hdl, "Message 1", websocketpp::frame::opcode::text);
        std::this_thread::sleep_for(std::chrono::seconds(2)); // 模拟延迟

        // 发送第二条消息
        s->send(hdl, "Message 2", websocketpp::frame::opcode::text);
        std::this_thread::sleep_for(std::chrono::seconds(2)); // 模拟延迟

        // 发送第三条消息
        s->send(hdl, "Message 3", websocketpp::frame::opcode::text);
    }).detach(); // 分离线程，避免阻塞主线程
}
int main() {
    // 创建服务器实例
    server echo_server;

    try {
        // 设置日志级别
        echo_server.set_access_channels(websocketpp::log::alevel::all);
        echo_server.clear_access_channels(websocketpp::log::alevel::frame_payload);

        // 初始化 ASIO
        echo_server.init_asio();

        // 注册连接打开时的回调函数
        echo_server.set_open_handler(bind(&on_open, &echo_server, ::_1));
        echo_server.set_message_handler(bind(&on_message,&echo_server,::_1,::_2));
        // 监听端口 9002
        echo_server.listen(9002);

        // 开始接受连接
        echo_server.start_accept();

        // 启动发送消息的线程
        std::thread sender_thread(send_messages, &echo_server);
        

        // 运行 ASIO 事件循环
        echo_server.run();

        // 等待发送线程结束
        sender_thread.join();
    } catch (websocketpp::exception const & e) {
        std::cout << "WebSocket++ exception: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Other exception" << std::endl;
    }

    return 0;
}