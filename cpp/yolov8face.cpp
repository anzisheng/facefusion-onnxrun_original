#include "yolov8face.h"

using namespace cv;
using namespace std;
using namespace Ort;

Yolov8Face::Yolov8Face(string model_path, const float conf_thres, const float iou_thresh)
{
    /// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注�?

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    /// std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
    /// ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法

    size_t numInputNodes = ort_session->GetInputCount();
    //cout << "numInputNodes = " << numInputNodes <<endl;
    size_t numOutputNodes = ort_session->GetOutputCount();
    //cout << "numOutputNodes = " << numOutputNodes <<endl;
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
        //input_names.push_back(ort_session->GetInputName(i, allocator));      ///低版本onnxruntime的接口函�?
        //input_names.push_back(ort_session->GetInputNameAllocated(i, allocator));      ///低版本onnxruntime的接口函�?
        AllocatedStringPtr input_name_Ptr = ort_session->GetInputNameAllocated(i, allocator);  /// 高版本onnxruntime的接口函�?
        //cout <<"input_name_Ptr.get():" << input_name_Ptr.get()<<endl;
        //input_names.push_back(input_name_Ptr.get()); /// 高版本onnxruntime的接口函�?
        input_names.push_back("images"); /// 高版本onnxruntime的接口函�?

        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        //output_names.push_back(ort_session->GetOutputName(i, allocator));  ///低版本onnxruntime的接口函�?
        AllocatedStringPtr output_name_Ptr= ort_session->GetOutputNameAllocated(i, allocator);
        //cout << "output=" << output_name_Ptr.get()<<endl;
        //output_names.push_back(output_name_Ptr.get()); /// 高版本onnxruntime的接口函�?
        output_names.push_back("output0"); /// 高版本onnxruntime的接口函�?

        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

    this->input_height = input_node_dims[0][2];
    this->input_width = input_node_dims[0][3];
    this->conf_threshold = conf_thres;
    this->iou_threshold = iou_thresh;
    this->CLASS_NAMES =  {"person"};    
}
void Yolov8Face::preprocess(Mat srcimg)
{
    const int height = srcimg.rows;
    const int width = srcimg.cols;
    Mat temp_image = srcimg.clone();
    if (height > this->input_height || width > this->input_width)
    {
        const float scale = std::min((float)this->input_height / height, (float)this->input_width / width);
        Size new_size = Size(int(width * scale), int(height * scale));
        resize(srcimg, temp_image, new_size);
    }
    this->ratio_height = (float)height / temp_image.rows;
    this->ratio_width = (float)width / temp_image.cols;
    Mat input_img;
    copyMakeBorder(temp_image, input_img, 0, this->input_height - temp_image.rows, 0, this->input_width - temp_image.cols, BORDER_CONSTANT, 0);

    vector<cv::Mat> bgrChannels(3);
    split(input_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }

    const int image_area = this->input_height * this->input_width;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}
/*


//<<<
*/
void Yolov8Face::postprocessDetect(Mat &log_img, float *pdata,std::vector<Bbox> &boxes, int num_box, int channels,  std::string file)
{    
    //std::cout << "num_box is : " << num_box <<std::endl;
    vector<Bbox> bounding_box_raw;
    vector<float> score_raw;
    for (int i = 0; i < num_box; i++)
    {
        const float score = pdata[4 * num_box + i];
        if (score > this->conf_threshold)
        {
            float xmin = (pdata[i] - 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (pdata[num_box + i] - 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (pdata[i] + 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (pdata[num_box + i] + 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            ////坐标的越界检查保护，可以添加一�?
            bounding_box_raw.emplace_back(Bbox{xmin, ymin, xmax, ymax});
            score_raw.emplace_back(score);
            /// 剩下�?5个关键点坐标的计�?,暂时不写,因为在下游的模块里没有用�?5个关键点坐标信息
        }
    }
    vector<int> keep_inds = nms(bounding_box_raw, score_raw, this->iou_threshold);
    const int keep_num = keep_inds.size();
    std::cout << "keep num " <<keep_num <<std::endl;
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i] = bounding_box_raw[ind];
    }
    cv::Scalar color = cv::Scalar(1, 1, 1);

    if(boxes.size() > 0)
    {          
        //cv::Rect rect(cv::Point(boxes[0].xmin, boxes[0].ymin), float(boxes[0].xmax - boxes[0].xmin), float(boxes[0].ymax -boxes[0].ymin));
        
        // int x = boxes[0].xmin;  
        // int y = boxes[0].ymin;  
        // int width = boxes[0].xmax - boxes[0].xmin;  
        // int height= boxes[0].ymax - boxes[0].ymin;  
        // cv::Rect rect(x, y, width, height);
        // cv::rectangle(srcimg, rect, cv::Scalar(0, 0, 255) ,4);

        for (size_t i = 0; i < keep_num; i++)
        {
            /* code */
            int x = boxes[i].xmin;  
            int y = boxes[i].ymin;  
            int width = boxes[i].xmax - boxes[i].xmin;  
            int height= boxes[i].ymax - boxes[i].ymin;  
            cv::Rect rect(x, y, width, height);
            cv::rectangle(log_img, rect, (i == 0) ? cv::Scalar(0, 0, 255):cv::Scalar(0, 0, 0) ,4);
            
        }
    }
    // if(photo)
    // {
    //     int pos = file.find_last_of('/');
    //     cout << "pos of photo issss:::: " << pos <<endl;
    //     std::string path_photo(file.substr(0, pos));


    //     imwrite(path_photo+"/log.jpg", log_img);
    // }


}
std::vector<Object>  Yolov8Face::postprocessDetect_gpu( float* pData, int num_box, int channels,  std::string file)
{

    auto numChannels = channels;//outputDims[0].d[1]; //20
    auto numAnchors = num_box;//outputDims[0].d[2]; //8400

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    //m_imgHeight = rgbMat.rows;
    //m_imgWidth = rgbMat.cols;
    //m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, pData);
    output = output.t();

    int numClasses = 1;
    float PROBABILITY_THRESHOLD = 0.25f;

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > PROBABILITY_THRESHOLD) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }


// Run NMS
    cv::dnn::NMSBoxes(bboxes, scores, /*labels, PROBABILITY_THRESHOLD*/ 0.25, /*NMS_THRESHOLD*/0.65, indices);
    std::cout << "indices size :" << indices.size()<<std::endl;
    std::vector<Object> objects;
    int TOP_K = 100;
    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices) {
        if (cnt >= TOP_K) {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
        std::cout << "cnt = " <<cnt <<std::endl;
    }



    return objects;




}





////只返回检测框,因为在下游的模块�?,置信度和5个关键点这两个信息在后续的模块里没有用到

void Yolov8Face::detect(Mat srcimg, std::vector<Object> &boxes, std::string file, bool photo)
{
    //cout <<"Yolov8Face::detect 000" <<endl;
    Mat log_img = srcimg.clone();

    this->preprocess(srcimg);

    this->m_imgHeight = srcimg.rows;
    this->m_imgWidth =  srcimg.cols;
    this->m_ratio = 1.f / std::min(640/ static_cast<float>(srcimg.cols), 640 / static_cast<float>(srcimg.rows));



    //cout <<"Yolov8Face::detect 111: "<<this->input_height<<" ,"<< this->input_width<<endl;
    std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());
    //cout <<"Yolov8Face::detect 222, input_tensor_ = "<< input_tensor_ <<endl;
    Ort::RunOptions runOptions;
    //cout <<"Yolov8Face::detect 222::" << this->input_names.data()<< "  "<<input_names[0]<<endl;
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());
    //cout <<"Yolov8Face::detect 333" <<endl;
    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状�?(1, 20, 8400),不考虑�?0维batchsize，每一列的长度20,�?4个元素是检测框坐标(cx,cy,w,h)，第4个元素是置信度，剩下�?15个元素是5个关键点坐标x,y和置信度
    const int num_box = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    int channels = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    //postprocessDetect(log_img, pdata, boxes, num_box,channels, file);
    //std::vector<Object> ret;
    boxes = postprocessDetect_gpu(pdata, num_box, channels, file);

    std::cout << "hellollllllllllllllllllll"<<endl;
    //std::cout << "ret size:" <<ret.size() << std::endl;
    std::cout << "hellol22222222222222222lll"<<endl;

    /*
    std::cout << "num_box is : " << num_box <<std::endl;
    vector<Bbox> bounding_box_raw;
    vector<float> score_raw;
    for (int i = 0; i < num_box; i++)
    {
        const float score = pdata[4 * num_box + i];
        if (score > this->conf_threshold)
        {
            float xmin = (pdata[i] - 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (pdata[num_box + i] - 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (pdata[i] + 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (pdata[num_box + i] + 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            ////坐标的越界检查保护，可以添加一�?
            bounding_box_raw.emplace_back(Bbox{xmin, ymin, xmax, ymax});
            score_raw.emplace_back(score);
            /// 剩下�?5个关键点坐标的计�?,暂时不写,因为在下游的模块里没有用�?5个关键点坐标信息
        }
    }
    vector<int> keep_inds = nms(bounding_box_raw, score_raw, this->iou_threshold);
    const int keep_num = keep_inds.size();
    std::cout << "keep num " <<keep_num <<std::endl;
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i] = bounding_box_raw[ind];
    }
    cv::Scalar color = cv::Scalar(1, 1, 1);

    if(boxes.size() > 0)
    {          
        //cv::Rect rect(cv::Point(boxes[0].xmin, boxes[0].ymin), float(boxes[0].xmax - boxes[0].xmin), float(boxes[0].ymax -boxes[0].ymin));
        
        // int x = boxes[0].xmin;  
        // int y = boxes[0].ymin;  
        // int width = boxes[0].xmax - boxes[0].xmin;  
        // int height= boxes[0].ymax - boxes[0].ymin;  
        // cv::Rect rect(x, y, width, height);
        // cv::rectangle(srcimg, rect, cv::Scalar(0, 0, 255) ,4);

        for (size_t i = 0; i < keep_num; i++)
        {
            
            int x = boxes[i].xmin;  
            int y = boxes[i].ymin;  
            int width = boxes[i].xmax - boxes[i].xmin;  
            int height= boxes[i].ymax - boxes[i].ymin;  
            cv::Rect rect(x, y, width, height);
            cv::rectangle(log_img, rect, (i == 0) ? cv::Scalar(0, 0, 255):cv::Scalar(0, 0, 0) ,4);
            
        }
    }*/
    //if(photo)
    drawObjectLabels(log_img, boxes, 2);
    std::cout << "hello99999999l"<<endl;
     {
        int pos = file.find_last_of('/');
        cout << "pos of photo issss:::: " << pos <<endl;
        std::string path_photo(file.substr(0, pos));


        imwrite(path_photo+"/log.jpg", log_img);
    }
}
void Yolov8Face::drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale) {
    // If segmentation information is present, start with that
    std::cout << "object size: "<< objects.size() <<std::endl;
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto &object : objects) {
            // Choose the color
            //int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
            cv::Scalar color = cv::Scalar(1, 1, 1);

            // Add the mask for said object
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    int index = 0;
    for (auto &object : objects) {

        // Choose the color
        //int colorIndex = object.label % .size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(1, 1, 1);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5) {
            txtColor = cv::Scalar(0, 0, 0);
        } else {
            txtColor = cv::Scalar(255,  255);
        }

        const auto &rect = object.rect;

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %.1f%%", "person", object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;
        //std::cout <<"box by trt is "<<x<< "  "<<y<<"  " << labelSize.width<<"  " << labelSize.height + baseLine << std::endl;
        
#ifdef SHOW
        
        cv::rectangle(image, rect, color * 255, scale + 1);
        std::cout << "draw......" << std::endl;
       

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), (index == 0) ? cv::Scalar(0, 0, 255):txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txt_bk_color , scale);
#endif
        // Pose estimation
        index += 1;
        
    }
}






