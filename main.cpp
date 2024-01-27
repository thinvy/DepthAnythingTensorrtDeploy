#include "depth_anything_trtruntime/trt_module.h"
#include <opencv2/opencv.hpp>

int main() {
    // 替换为你的视频文件路径
    std::string video_path = "../data/pocket3_night.mp4";
    TRTModule model("../weights/depth_anything_vits14-sim-ptq-f16.plan");
    // TRTModule model("../weights/depth_anything_vits14.engine");

    cv::VideoCapture cap(video_path);

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat colored_depth;

    cv::namedWindow("depth anything", cv::WINDOW_NORMAL);
    cv::resizeWindow("depth anything", cv::Size(960, 1080));


    while (true) {
        cap >> frame;  // 读取一帧

        // 检查是否到达视频末尾
        if (frame.empty()) {
            std::cout << "End of video." << std::endl;
            break;
        }

        // 在这里添加你的 TensorRT 模型的推理代码
        // std::cout << "start infer" << std::endl;
        cv::Mat depth = model.predict(frame);
        // std::cout << "finish infer" << std::endl;

        // 将深度图应用颜色映射
        cv::applyColorMap(depth, colored_depth, cv::COLORMAP_INFERNO);

        // 显示当前帧和深度图
        cv::Mat showImage;
        cv::vconcat(frame, colored_depth, showImage);

        cv::imshow("depth anything", showImage);

        // 按 ESC 键退出循环
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 关闭视频流
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
