/**
 * @file creat_points
 * @author eunsoo
 * @brief customizing cam_lidar_calib package
 * @version 0.1
 * @date 2022-05-19
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once
#include <custom_optimize.h>
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
#include <opencv2/opencv.hpp>
#include <random>

#include "ceres/ceres.h"
#include "ceres/covariance.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "open3d_ros/open3d_ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> SyncPolicy;

class customCalib {
   private:
    ros::NodeHandle nh_;
    ros::Publisher cloud_pub_, pcd_pub_;
    ros::Subscriber cloud_sub_;

    std::vector<pointsPair> points_pair_;
    cv::Mat projection_matrix_;
    cv::Mat distCoeff_;
    cv::Mat tvec_, rvec_;
    cv::Mat C_R_W_;
    Eigen::Matrix3d c_R_w_;
    Eigen::Vector3d c_t_w_;
    std::vector<cv::Point3f> objectPoints_;  // 3d world coordinates
    std::vector<cv::Point2f> imagePoints_;   // 2d image coordinates
    std::shared_ptr<open3d::geometry::PointCloud> plane_cloud_;

    std::string result_str_, result_json_, load_pcd_path_;
    std::string lidar_in_topic_;
    std::string file_type_;

    int image_width_, image_height_;
    double x_min_, x_max_;
    double y_min_, y_max_;
    double z_min_, z_max_;
    double ransac_threshold_;
    int opt_cnt_ = 1;
    std::string cam_config_file_path_;
    int num_data_;

    void readCameraParams(std::string cam_config_file_path_, int &image_height_, int &image_width_, cv::Mat &D, cv::Mat &K);
    void getParamFunc(ros::NodeHandle &priv_nh_);
    void writeResultToJson(Eigen::MatrixXd &C_T_L, Eigen::Matrix3d &Rotn);
    void handleLidarWithOpen3D(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool solveOpt();
    void createMatchingPoints(std::shared_ptr<open3d::geometry::PointCloud> pcd);
    void makeNoise();
    void findInitialValue(Eigen::Vector6d &opt_param);

   public:
    int returnNumOfData() {
        return num_data_;
    }
    void publishPCD(sensor_msgs::PointCloud2 cloud_msgs) {
        ros::Time time = ros::Time::now();
        cloud_msgs.header.stamp = time;
        pcd_pub_.publish(cloud_msgs);
    }

    sensor_msgs::PointCloud2 converPCDtoROS(auto &pcd) {
        sensor_msgs::PointCloud2 ros_pc2;
        open3d_ros::open3dToRos(pcd, ros_pc2, "o3d_frame");
        return ros_pc2;
    }

    std::string getFiletype() {
        return file_type_;
    }
    std::string getPcdfilePath() {
        return load_pcd_path_;
    }
    customCalib(ros::NodeHandle &priv_nh_) {
        getParamFunc(priv_nh_);
        c_R_w_ = Eigen::Matrix3d::Zero();
        c_t_w_ = Eigen::Vector3d::Zero();
        cloud_sub_ = nh_.subscribe(lidar_in_topic_, 10, &customCalib::handleLidarWithOpen3D, this);

        pcd_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_in_topic_, 10);
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("points_out", 1);
        projection_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff_ = cv::Mat::zeros(5, 1, CV_64F);
        // boardDetectedInCam_ = false;
        tvec_ = cv::Mat::zeros(3, 1, CV_64F);
        rvec_ = cv::Mat::zeros(3, 1, CV_64F);
        C_R_W_ = cv::Mat::eye(3, 3, CV_64F);
        c_R_w_ = Eigen::Matrix3d::Identity();

        std::ifstream myReadFile(result_str_.c_str());
        std::string word;
        int i = 0;
        int j = 0;
        Eigen::Matrix4d C_T_L, L_T_C;
        Eigen::Matrix3d C_R_L, L_R_C;
        Eigen::Quaterniond C_R_L_quatn, L_R_C_quatn;
        Eigen::Vector3d C_t_L, L_t_C;
        cv::Mat c_R_l;
        while (myReadFile >> word) {
            C_T_L(i, j) = atof(word.c_str());
            j++;
            if (j > 3) {
                j = 0;
                i++;
            }
        }
        L_T_C = C_T_L.inverse();  // cam to lidar

        C_R_L = C_T_L.block(0, 0, 3, 3);
        C_t_L = C_T_L.block(0, 3, 3, 1);

        L_R_C = L_T_C.block(0, 0, 3, 3);
        L_t_C = L_T_C.block(0, 3, 3, 1);

        cv::eigen2cv(C_R_L, c_R_l);
        C_R_L_quatn = Eigen::Quaterniond(C_R_L);
        L_R_C_quatn = Eigen::Quaterniond(L_R_C);
        cv::Rodrigues(c_R_l, rvec_);
        cv::eigen2cv(C_t_L, tvec_);
        readCameraParams(cam_config_file_path_, image_height_, image_width_, distCoeff_, projection_matrix_);
        std::cout << "proj : " << std::endl
                  << projection_matrix_ << std::endl;
        std::cout << "distCoeff : " << std::endl
                  << distCoeff_ << std::endl;
        std::cout << "result_str_ : " << std::endl
                  << C_R_L << std::endl;
        ROS_INFO("init finish");
    }
    ~customCalib();
};

customCalib::~customCalib() {
    ROS_INFO("terminate");
}
