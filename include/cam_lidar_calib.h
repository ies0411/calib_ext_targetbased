/**
 * @file cam_lidar_calib_node.cpp
 * @author eunsoo
 * @brief customizing cam_lidar_calib package
 * @version 0.1
 * @date 2022-05-19
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once
#include <calibration_error_term.h>
#include <cv_bridge/cv_bridge.h>
#include <jsoncpp/json/config.h>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <open3d/Open3D.h>
#include <open3d/io/PointCloudIO.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/Image.h>
#include <string.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <random>

#include "ceres/ceres.h"
#include "ceres/covariance.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "open3d_ros/open3d_ros.h"
#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
// TODO : rosbag, file 구분하여 처리, projection 코드 확인

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> SyncPolicy;
enum POINT_DIRECTION {
    X_,
    Y_,
    Z_,
};

class camLidarCalib {
   private:
    ros::NodeHandle nh_;
    ros::Publisher cloud_pub_, pcd_pub_, img_pub_;
    ros::Subscriber open3d_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub_;
    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Synchronizer<SyncPolicy> *sync_;
    cv::Mat image_in_;
    cv::Mat image_resized_;
    cv::Mat projection_matrix_;
    cv::Mat distCoeff_;
    std::vector<cv::Point2f> image_points_;
    std::vector<cv::Point3f> object_points_;
    std::vector<cv::Point2f> projected_points_;
    bool boardDetectedInCam_;
    double dx_, dy_;
    int checkerboard_rows_, checkerboard_cols_;
    int min_points_on_plane_;
    cv::Mat tvec_, rvec_;
    cv::Mat C_R_W_;
    Eigen::Matrix3d c_R_w_;
    Eigen::Vector3d c_t_w_;
    Eigen::Vector3d r3_;
    Eigen::Vector3d r3_old_;
    Eigen::Vector3d Nc_;
    std::shared_ptr<open3d::geometry::PointCloud> plane_cloud_;
    std::vector<Eigen::Vector3d> lidar_points_;
    std::vector<std::vector<Eigen::Vector3d> > all_lidar_points_;
    // std::vector<Eigen::Vector3d> lidar_points_;
    std::vector<Eigen::Vector3d> all_normals_;
    bool filtering_ = true;
    std::string result_str_, result_json_, load_img_path_, load_pcd_path_;
    // result_rpy_
    std::string camera_in_topic_;
    std::string lidar_in_topic_;
    std::string file_type_;
    std::vector<open3d::geometry::PointCloud> all_points_;
    int num_views_;
    bool show_debug_ = true;
    std::string cam_config_file_path_;
    int image_width_, image_height_;
    double x_min_, x_max_;
    double y_min_, y_max_;
    double z_min_, z_max_;
    double ransac_threshold_ = 0.01;

    int num_data_;
    // std::string initializations_file_;
    std::ofstream init_file_;

    void readCameraParams(std::string cam_config_file_path_, int &image_height_, int &image_width_, cv::Mat &D, cv::Mat &K);
    bool imageHandler(const sensor_msgs::ImageConstPtr &image_msg);
    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, const sensor_msgs::ImageConstPtr &image_msg);
    void getParamFunc(ros::NodeHandle &priv_nh_);
    void writeResultToJson(Eigen::MatrixXd &C_T_L, Eigen::Matrix3d &Rotn);
    // void open3DCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool handleLidarWithOpen3D(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool solveWithOpen3D();
    bool cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool solveWithPCL();

   public:
    double publish_rate_ = 10;
    int data_size_ = 0;
    std::string img_type_ = "png";
    sensor_msgs::Image covertCVtoROS(cv::Mat &frame) {
        sensor_msgs::Image ros_img;
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg(ros_img);
        return ros_img;
    }
    int returnNumOfData() {
        return num_data_;
    }
    void publishImgAndPCD(sensor_msgs::Image img_msgs, sensor_msgs::PointCloud2 cloud_msgs) {
        // img_msgs.Header.stamp = 1;
        ros::Time time = ros::Time::now();
        cloud_msgs.header.stamp = time;
        img_msgs.header.stamp = time;
        img_pub_.publish(img_msgs);
        pcd_pub_.publish(cloud_msgs);
    }

    sensor_msgs::PointCloud2 converPCDtoROS(auto &pcd) {
        sensor_msgs::PointCloud2 ros_pc2;
        open3d_ros::open3dToRos(pcd, ros_pc2, "o3d_frame");
        return ros_pc2;
    }

    std::string getImagefilePath() {
        return load_img_path_;
    }

    std::string getFiletype() {
        return file_type_;
    }
    std::string getPcdfilePath() {
        return load_pcd_path_;
    }
    camLidarCalib(ros::NodeHandle &priv_nh_) {
        getParamFunc(priv_nh_);
        c_R_w_ = Eigen::Matrix3d::Zero();
        c_t_w_ = Eigen::Vector3d::Zero();
        r3_ = Eigen::Vector3d::Zero();
        r3_old_ = Eigen::Vector3d::Zero();
        Nc_ = Eigen::Vector3d::Zero();

        cloud_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, lidar_in_topic_, 10);
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh_, camera_in_topic_, 10);
        sync_ = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(100), *cloud_sub_, *image_sub);
        sync_->registerCallback(boost::bind(&camLidarCalib::callback, this, _1, _2));
        // open3d_sub_ = nh_.subscribe(lidar_in_topic_, 10, &camLidarCalib::open3DCallback, this);

        pcd_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_in_topic_, 10);
        img_pub_ = nh_.advertise<sensor_msgs::Image>(camera_in_topic_, 10);
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("points_out", 1);
        projection_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff_ = cv::Mat::zeros(5, 1, CV_64F);
        boardDetectedInCam_ = false;
        tvec_ = cv::Mat::zeros(3, 1, CV_64F);
        rvec_ = cv::Mat::zeros(3, 1, CV_64F);
        C_R_W_ = cv::Mat::eye(3, 3, CV_64F);
        c_R_w_ = Eigen::Matrix3d::Identity();

        // set real checkerboard points
        for (int i = 0; i < checkerboard_rows_; i++)
            for (int j = 0; j < checkerboard_cols_; j++)
                object_points_.emplace_back(cv::Point3f(j * dy_, i * dx_, 0.0));
        readCameraParams(cam_config_file_path_, image_height_, image_width_, distCoeff_, projection_matrix_);
        std::cout << "intinsic : " << std::endl
                  << projection_matrix_ << std::endl;
        std::cout << "distCoeff_ : " << std::endl
                  << distCoeff_ << std::endl;
        std::cout << "image_width_ : " << std::endl
                  << image_width_ << std::endl;
        std::cout << "image_height_ : " << std::endl
                  << image_height_ << std::endl;
        ROS_INFO("init finish");
    }
    ~camLidarCalib();
};

camLidarCalib::~camLidarCalib() {
    ROS_INFO("terminate");
}
