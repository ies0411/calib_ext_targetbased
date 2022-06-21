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
#include "open3d_visualizer.h"
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
    std::vector<Eigen::Vector3d> all_normals_;

    std::string result_str_, result_rpy_, result_json_, load_img_path_, load_pcd_path_;

    std::string camera_in_topic_;
    std::string lidar_in_topic_;
    std::string file_type_;
    std::vector<open3d::geometry::PointCloud> all_points_;
    int num_views_;

    std::string cam_config_file_path_;
    int image_width_, image_height_;
    double x_min_, x_max_;
    double y_min_, y_max;
    double z_min_, z_max_;
    double ransac_threshold_;
    int no_of_initializations_;
    std::string initializations_file_;
    std::ofstream init_file_;

    void readCameraParams(std::string cam_config_file_path_, int &image_height_, int &image_width_, cv::Mat &D, cv::Mat &K);
    bool imageHandler(const sensor_msgs::ImageConstPtr &image_msg);
    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, const sensor_msgs::ImageConstPtr &image_msg);
    void getParamFunc(ros::NodeHandle &priv_nh_);
    void writeResultToJson(Eigen::MatrixXd &C_T_L, Eigen::Matrix3d &Rotn);
    // void open3DCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool handleLidarWithOpen3D(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool solveWithOpen3D();

   public:
    sensor_msgs::Image covertCVtoROS(cv::Mat &frame) {
        sensor_msgs::Image ros_img;
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg(ros_img);
        return ros_img;
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

        if (file_type_ == "rosbag") {
            cloud_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, lidar_in_topic_, 10);
            image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh_, camera_in_topic_, 10);
            sync_ = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(100), *cloud_sub_, *image_sub);
            sync_->registerCallback(boost::bind(&camLidarCalib::callback, this, _1, _2));
            // open3d_sub_ = nh_.subscribe(lidar_in_topic_, 10, &camLidarCalib::open3DCallback, this);
        } else if (file_type_ == "file") {
        } else {
            ROS_WARN("file_type param name error");
        }
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
                object_points_.emplace_back(cv::Point3f(i * dx_, j * dy_, 0.0));
        readCameraParams(cam_config_file_path_, image_height_, image_width_, distCoeff_, projection_matrix_);
        ROS_INFO("init finish");
    }
    ~camLidarCalib();
};

camLidarCalib::~camLidarCalib() {
    ROS_INFO("terminate");
}

bool camLidarCalib::handleLidarWithOpen3D(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    std::shared_ptr<open3d::geometry::PointCloud> pcd = std::make_shared<open3d::geometry::PointCloud>();

    open3d_ros::rosToOpen3d(cloud_msg, *pcd);

    std::shared_ptr<open3d::geometry::PointCloud> filtered_pcd = std::make_shared<open3d::geometry::PointCloud>();
    filtered_pcd->points_.clear();
    for (size_t i = 0; i < pcd->points_.size(); i++) {
        const Eigen::Vector3d tmp_point = pcd->points_[i];
        if (tmp_point[X_] > x_min_ && tmp_point[X_] < x_max_ &&
            tmp_point[Y_] > y_min_ && tmp_point[Y_] < y_max &&
            tmp_point[Z_] > z_min_ && tmp_point[Z_] < z_max_) {
            filtered_pcd->points_.push_back(tmp_point);
        }
    }

    auto [plane_model, plane_points] = filtered_pcd->SegmentPlane(0.005, 5, 100);

    plane_cloud_ = filtered_pcd->SelectByIndex(plane_points);

    sensor_msgs::PointCloud2 out_cloud;
    open3d_ros::open3dToRos(*plane_cloud_, out_cloud, "o3d_frame");
    cloud_pub_.publish(out_cloud);
    if (plane_cloud_->points_.size() < min_points_on_plane_) return false;
    return true;
}
bool camLidarCalib::solveWithOpen3D() {
    if (r3_.dot(r3_old_) > 0.95) return false;
    r3_old_ = r3_;
    all_normals_.push_back(Nc_);
    all_points_.push_back(plane_cloud_->points_);
    ROS_INFO("num of view : %d", all_normals_.size());
    if (all_normals_.size() < num_views_) return false;
    ROS_INFO_STREAM("Starting optimization...");
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();
    Eigen::Vector3d axis_angle;
    ceres::RotationMatrixToAngleAxis(rotation_matrix.data(), axis_angle.data());  //  convert to Rodrigus type
    Eigen::Vector6d opt_param;
    opt_param(0) = axis_angle(0);
    opt_param(1) = axis_angle(1);
    opt_param(2) = axis_angle(2);
    opt_param(3) = translation_vector(0);
    opt_param(4) = translation_vector(1);
    opt_param(5) = translation_vector(2);

    ceres::LossFunction *loss_function = NULL;
    ceres::Problem problem;
    problem.AddParameterBlock(opt_param.data(), opt_param.size());

    for (int i = 0; i < (int)all_normals_.size(); i++) {
        Eigen::Vector3d normal_i = all_normals_[i];
        std::vector<Eigen::Vector3d> lidar_point_i = all_points_[i].points_;
        for (int j = 0; j < lidar_point_i.size(); j++) {
            Eigen::Vector3d lidar_point = lidar_point_i[j];
            ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CalibrationErrorTerm, 1, 6>(new CalibrationErrorTerm(lidar_point, normal_i));
            problem.AddResidualBlock(cost_function, loss_function, opt_param.data());
        }
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 300;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    axis_angle(0) = opt_param(0);
    axis_angle(1) = opt_param(1);
    axis_angle(2) = opt_param(2);
    ceres::AngleAxisToRotationMatrix(axis_angle.data(), rotation_matrix.data());
    translation_vector(0) = opt_param(3);
    translation_vector(1) = opt_param(4);
    translation_vector(2) = opt_param(5);
    std::ofstream results;
    results.open(result_str_);
    results << "rotation\n"
            << rotation_matrix
            << "\ntranslation\n"
            << translation_vector;
    results.close();
    ROS_INFO_STREAM("complete..");
    ros::shutdown();
}
void camLidarCalib::getParamFunc(ros::NodeHandle &priv_nh_) {
    priv_nh_.param<std::string>("file_type", file_type_, "rosbag");
    priv_nh_.param<double>("dx_", dx_, 0.075);
    priv_nh_.param<double>("dy_", dy_, 0.075);
    priv_nh_.param<int>("checkerboard_rows_", checkerboard_rows_, 9);
    priv_nh_.param<int>("checkerboard_cols_", checkerboard_cols_, 6);

    priv_nh_.param<int>("min_points_on_plane_", min_points_on_plane_, 450);
    priv_nh_.param<int>("num_views_", num_views_, 10);
    priv_nh_.param<std::string>("initializations_file_", initializations_file_, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/init.txt"));

    priv_nh_.param<int>("no_of_initializations_", no_of_initializations_, 1);
    priv_nh_.param<std::string>("json_result_file", result_json_, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/result_json.json"));
    priv_nh_.param<std::string>("load_path", load_img_path_, std::string("/home/catkin_ws/src/"));
    priv_nh_.param<std::string>("result_file", result_str_, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/CTL.txt"));
    priv_nh_.param<std::string>("result_rpy_file", result_rpy_, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/rpy.txt"));
    priv_nh_.param<std::string>("cam_config_file_path_", cam_config_file_path_, std::string("/home/catkin_ws/src/cam_lidar_calib/config/basler_config.yaml"));

    priv_nh_.param<double>("x_min_", x_min_, 0);
    priv_nh_.param<double>("x_max_", x_max_, 6);
    priv_nh_.param<double>("y_min_", y_min_, -1.25);
    priv_nh_.param<double>("y_max", y_max, 1.25);
    priv_nh_.param<double>("z_min_", z_min_, -0.5);
    priv_nh_.param<double>("z_max_", z_max_, 2.0);
    priv_nh_.param<double>("ransac_threshold_", ransac_threshold_, 0.01);

    priv_nh_.param<std::string>("camera_in_topic_", camera_in_topic_, std::string("/pylon_camera_node/image_raw"));
    priv_nh_.param<std::string>("lidar_in_topic_", lidar_in_topic_, std::string("/velodyne_points"));
}

void camLidarCalib::readCameraParams(std::string cam_config_file_path_, int &image_height_, int &image_width_, cv::Mat &D, cv::Mat &K) {
    cv::FileStorage fs_cam_config(cam_config_file_path_, cv::FileStorage::READ);
    if (!fs_cam_config.isOpened())
        std::cerr << "Error: Wrong path: " << cam_config_file_path_ << std::endl;
    fs_cam_config["image_height_"] >> image_height_;
    fs_cam_config["image_width_"] >> image_width_;
    fs_cam_config["k1"] >> D.at<double>(0);
    fs_cam_config["k2"] >> D.at<double>(1);
    fs_cam_config["p1"] >> D.at<double>(2);
    fs_cam_config["p2"] >> D.at<double>(3);
    fs_cam_config["k3"] >> D.at<double>(4);
    fs_cam_config["fx"] >> K.at<double>(0, 0);
    fs_cam_config["fy"] >> K.at<double>(1, 1);
    fs_cam_config["cx"] >> K.at<double>(0, 2);
    fs_cam_config["cy"] >> K.at<double>(1, 2);
}

bool camLidarCalib::imageHandler(const sensor_msgs::ImageConstPtr &image_msg) {
    try {
        // ros type -> cv type
        image_in_ = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        if (-1 == cv::findChessboardCorners(image_in_, cv::Size(checkerboard_cols_, checkerboard_rows_), image_points_, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE)) {
            return false;
        }

        cv::drawChessboardCorners(image_in_, cv::Size(checkerboard_cols_, checkerboard_rows_), image_points_, boardDetectedInCam_);
        // std::cout << image_points_.size() << "==" << object_points_.size() << std::endl;
        if (image_points_.size() == object_points_.size()) {
            cv::solvePnP(object_points_, image_points_, projection_matrix_, distCoeff_, rvec_, tvec_, false, CV_ITERATIVE);
            projected_points_.clear();
            cv::projectPoints(object_points_, rvec_, tvec_, projection_matrix_, distCoeff_, projected_points_, cv::noArray());
            // std::cout << projected_points_.size() << std::endl;
            for (int i = 0; i < projected_points_.size(); i++) {
                cv::circle(image_in_, projected_points_[i], 16, cv::Scalar(0, 255, 0), 10, cv::LINE_AA, 0);
            }
            cv::Rodrigues(rvec_, C_R_W_);
            cv::cv2eigen(C_R_W_, c_R_w_);
            c_t_w_ = Eigen::Vector3d(tvec_.at<double>(0), tvec_.at<double>(1), tvec_.at<double>(2));
            r3_ = c_R_w_.block<3, 1>(0, 2);
            Nc_ = (r3_.dot(c_t_w_)) * r3_;
        }
        cv::resize(image_in_, image_resized_, cv::Size(), 0.25, 0.25);
        cv::imshow("view", image_resized_);
        cv::waitKey(10);
        return true;
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", image_msg->encoding.c_str());
        return false;
    }
}

void camLidarCalib::writeResultToJson(Eigen::MatrixXd &C_T_L, Eigen::Matrix3d &Rotn) {
    Json::Value extrin_homogeneous;
    for (int i = 0; i < C_T_L.rows(); i++) {
        for (int j = 0; j < C_T_L.cols(); j++) {
            extrin_homogeneous["Homogeneous"][i].append(C_T_L(i, j));
        }
    }
    Eigen::Vector3d angle = Rotn.eulerAngles(0, 1, 2) * 180 / M_PI;
    for (int i = 0; i < 3; i++) {
        std::cout << angle[i] << std::endl;
        extrin_homogeneous["rpy"].append(angle[i]);
    }
    Eigen::Vector3d translation = C_T_L.block(0, 3, 3, 1);
    for (int i = 0; i < 3; i++) {
        extrin_homogeneous["translation"].append(translation[i]);
    }

    Json::StyledWriter writer;
    auto str = writer.write(extrin_homogeneous);
    std::ofstream result_json_file(result_json_, std::ofstream::out | std::ofstream::trunc);
    result_json_file << str;
    result_json_file.close();
}

void camLidarCalib::callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, const sensor_msgs::ImageConstPtr &image_msg) {
    if (!imageHandler(image_msg)) return;
    if (!handleLidarWithOpen3D(cloud_msg)) return;
    solveWithOpen3D();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "CameraLidarCalib_node");
    ros::NodeHandle nh_("~");
    camLidarCalib cLC(nh_);

    if (cLC.getFiletype() == "rosbag") {
        ros::spin();
    } else if (cLC.getFiletype() == "file") {
        std::vector<cv::String> images;
        std::vector<open3d::geometry::PointCloud> pcds;
        cv::glob(cLC.getImagefilePath(), images);
        int size = images.size();
        ROS_INFO("number of images : %d", size);

        for (const auto &p : std::experimental::filesystem::recursive_directory_iterator(cLC.getPcdfilePath())) {
            auto pcd = std::make_shared<open3d::geometry::PointCloud>();
            open3d::io::ReadPointCloud(p.path(), *pcd);
        }
        if (images.size() != pcds.size()) {
            ROS_WARN("pcd's size is not equal to img's size");
            return -1;
        }

        u_int idx = 0;
        while (ros::ok()) {
            cv::Mat frame = cv::imread(images[idx]);
            auto pcd = pcds[idx++];

            cLC.publishImgAndPCD(cLC.covertCVtoROS(frame), cLC.converPCDtoROS(pcd));
            if (idx == size) {
                ROS_WARN("need more various img");
                return -1;
            }
        }
    }

    return 0;
}
