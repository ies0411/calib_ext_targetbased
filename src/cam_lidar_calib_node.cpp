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

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <ctime>
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

/*****************/

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> SyncPolicy;
enum POINT_DIRECTION {
    X_,
    Y_,
    Z_,
};

class camLidarCalib {
   private:
    ros::NodeHandle nh;
    ros::Publisher cloud_pub;
    ros::Subscriber open3d_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Synchronizer<SyncPolicy> *sync;
    cv::Mat image_in;
    cv::Mat image_resized;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> projected_points;
    bool boardDetectedInCam;
    double dx, dy;
    int checkerboard_rows, checkerboard_cols;
    int min_points_on_plane;
    cv::Mat tvec, rvec;
    cv::Mat C_R_W;
    Eigen::Matrix3d c_R_w;
    Eigen::Vector3d c_t_w;
    Eigen::Vector3d r3;
    Eigen::Vector3d r3_old;
    Eigen::Vector3d Nc;
    std::shared_ptr<o3d::geometry::PointCloud> plane_cloud;
    std::vector<Eigen::Vector3d> lidar_points;
    std::vector<std::vector<Eigen::Vector3d> > all_lidar_points;
    std::vector<Eigen::Vector3d> all_normals;

    std::string result_str, result_rpy, result_json_;

    std::string camera_in_topic;
    std::string lidar_in_topic;
    std::vector<o3d::geometry::PointCloud> all_points;
    int num_views;

    std::string cam_config_file_path;
    int image_width, image_height;
    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;
    double ransac_threshold;
    int no_of_initializations;
    std::string initializations_file;
    std::ofstream init_file;

    void readCameraParams(std::string cam_config_file_path, int &image_height, int &image_width, cv::Mat &D, cv::Mat &K);
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool imageHandler(const sensor_msgs::ImageConstPtr &image_msg);
    void runSolver();
    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, const sensor_msgs::ImageConstPtr &image_msg);
    void getParamFunc(ros::NodeHandle &priv_nh);
    void writeResultToJson(Eigen::MatrixXd &C_T_L, Eigen::Matrix3d &Rotn);
    // void open3DCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool handleLidarWithOpen3D(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    bool solveWithOpen3D();

   public:
    camLidarCalib(ros::NodeHandle &priv_nh) {
        getParamFunc(priv_nh);
        c_R_w = Eigen::Matrix3d::Zero();
        c_t_w = Eigen::Vector3d::Zero();
        r3 = Eigen::Vector3d::Zero();
        r3_old = Eigen::Vector3d::Zero();
        Nc = Eigen::Vector3d::Zero();

        cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, lidar_in_topic, 10);
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, camera_in_topic, 10);

        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(100), *cloud_sub, *image_sub);
        sync->registerCallback(boost::bind(&camLidarCalib::callback, this, _1, _2));
        // open3d_sub = nh.subscribe(lidar_in_topic, 10, &camLidarCalib::open3DCallback, this);
        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("points_out", 1);
        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(5, 1, CV_64F);
        boardDetectedInCam = false;
        tvec = cv::Mat::zeros(3, 1, CV_64F);
        rvec = cv::Mat::zeros(3, 1, CV_64F);
        C_R_W = cv::Mat::eye(3, 3, CV_64F);
        c_R_w = Eigen::Matrix3d::Identity();

        // set real checkerboard points
        for (int i = 0; i < checkerboard_rows; i++)
            for (int j = 0; j < checkerboard_cols; j++)
                object_points.emplace_back(cv::Point3f(i * dx, j * dy, 0.0));

        readCameraParams(cam_config_file_path, image_height, image_width, distCoeff, projection_matrix);

        ROS_INFO("init finish");
    }
    ~camLidarCalib();
};

camLidarCalib::~camLidarCalib() {
    ROS_INFO("terminate");
}

bool camLidarCalib::handleLidarWithOpen3D(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    // = std::make_shared<open3d::geometry::PointCloud>();
    std::shared_ptr<o3d::geometry::PointCloud> pcd;

    pcd = std::make_shared<open3d::geometry::PointCloud>();

    open3d_ros::rosToOpen3d(cloud_msg, *pcd);

    std::shared_ptr<o3d::geometry::PointCloud> filtered_pcd = std::make_shared<open3d::geometry::PointCloud>();
    filtered_pcd->points_.clear();
    for (size_t i = 0; i < pcd->points_.size(); i++) {
        const Eigen::Vector3d tmp_point = pcd->points_[i];
        if (tmp_point[X_] > x_min && tmp_point[X_] < x_max &&
            tmp_point[Y_] > y_min && tmp_point[Y_] < y_max &&
            tmp_point[Z_] > z_min && tmp_point[Z_] < z_max) {
            filtered_pcd->points_.push_back(tmp_point);
        }
    }

    auto [plane_model, plane_points] = filtered_pcd->SegmentPlane(0.005, 5, 100);  // TODO: ransac param 설정

    // plane_cloud->points_.clear();
    plane_cloud = filtered_pcd->SelectByIndex(plane_points);

    sensor_msgs::PointCloud2 out_cloud;
    open3d_ros::open3dToRos(*plane_cloud, out_cloud, "o3d_frame");
    cloud_pub.publish(out_cloud);

    if (plane_cloud->points_.size() < min_points_on_plane) return false;
    return true;
}
bool camLidarCalib::solveWithOpen3D() {
    if (r3.dot(r3_old) > 0.95) return false;
    r3_old = r3;
    all_normals.push_back(Nc);
    all_points.push_back(plane_cloud->points_);
    ROS_INFO("num of view : %d", all_normals.size());
    if (all_normals.size() < num_views) return false;
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

    for (int i = 0; i < (int)all_normals.size(); i++) {
        Eigen::Vector3d normal_i = all_normals[i];
        std::vector<Eigen::Vector3d> lidar_point_i = all_points[i].points_;
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
    results.open(result_str);
    results << "rotation\n"
            << rotation_matrix
            << "\ntranslation\n"
            << translation_vector;
    results.close();
    ROS_INFO_STREAM("complete..");
    ros::shutdown();
}
void camLidarCalib::getParamFunc(ros::NodeHandle &priv_nh) {
    priv_nh.param<double>("dx", dx, 0.075);
    priv_nh.param<double>("dy", dy, 0.075);
    priv_nh.param<int>("checkerboard_rows", checkerboard_rows, 9);
    priv_nh.param<int>("checkerboard_cols", checkerboard_cols, 6);

    priv_nh.param<int>("min_points_on_plane", min_points_on_plane, 450);
    priv_nh.param<int>("num_views", num_views, 10);
    priv_nh.param<std::string>("initializations_file", initializations_file, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/init.txt"));

    priv_nh.param<int>("no_of_initializations", no_of_initializations, 1);
    priv_nh.param<std::string>("json_result_file", result_json_, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/result_json.json"));

    priv_nh.param<std::string>("result_file", result_str, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/CTL.txt"));
    priv_nh.param<std::string>("result_rpy_file", result_rpy, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/rpy.txt"));
    priv_nh.param<std::string>("cam_config_file_path", cam_config_file_path, std::string("/home/catkin_ws/src/cam_lidar_calib/config/basler_config.yaml"));

    priv_nh.param<double>("x_min", x_min, 0);
    priv_nh.param<double>("x_max", x_max, 6);
    priv_nh.param<double>("y_min", y_min, -1.25);
    priv_nh.param<double>("y_max", y_max, 1.25);
    priv_nh.param<double>("z_min", z_min, -0.5);
    priv_nh.param<double>("z_max", z_max, 2.0);
    priv_nh.param<double>("ransac_threshold", ransac_threshold, 0.01);

    priv_nh.param<std::string>("camera_in_topic", camera_in_topic, std::string("/pylon_camera_node/image_raw"));
    priv_nh.param<std::string>("lidar_in_topic", lidar_in_topic, std::string("/velodyne_points"));
}

void camLidarCalib::readCameraParams(std::string cam_config_file_path, int &image_height, int &image_width, cv::Mat &D, cv::Mat &K) {
    cv::FileStorage fs_cam_config(cam_config_file_path, cv::FileStorage::READ);
    if (!fs_cam_config.isOpened())
        std::cerr << "Error: Wrong path: " << cam_config_file_path << std::endl;
    fs_cam_config["image_height"] >> image_height;
    fs_cam_config["image_width"] >> image_width;
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

void camLidarCalib::cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // ros pointclout type -> pcl type
    pcl::fromROSMsg(*cloud_msg, *in_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    /// Pass through filters
    pcl::PassThrough<pcl::PointXYZ> pass_x;
    pass_x.setInputCloud(in_cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(x_min, x_max);
    pass_x.filter(*cloud_filtered_x);

    pcl::PassThrough<pcl::PointXYZ> pass_y;
    pass_y.setInputCloud(cloud_filtered_x);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(y_min, y_max);
    pass_y.filter(*cloud_filtered_y);

    pcl::PassThrough<pcl::PointXYZ> pass_z;
    pass_z.setInputCloud(cloud_filtered_y);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(z_min, z_max);
    pass_z.filter(*cloud_filtered_z);

    /// Plane Segmentation
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_z));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
    ransac.setDistanceThreshold(ransac_threshold);
    ransac.computeModel();
    std::vector<int> inliers_indicies;
    ransac.getInliers(inliers_indicies);
    pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered_z, inliers_indicies, *plane);

    /// Statistical Outlier Removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(plane);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1);
    sor.filter(*plane_filtered);

    /// Store the points lying in the filtered plane in a vector
    lidar_points.clear();
    for (size_t i = 0; i < plane_filtered->points.size(); i++) {
        double X = plane_filtered->points[i].x;
        double Y = plane_filtered->points[i].y;
        double Z = plane_filtered->points[i].z;
        lidar_points.push_back(Eigen::Vector3d(X, Y, Z));
    }
    ROS_INFO_STREAM("No of planar_pts: " << lidar_points.size());
    // ROS_WARN_STREAM("No of planar_pts: " << plane_filtered->points.size());
    sensor_msgs::PointCloud2 out_cloud;
    pcl::toROSMsg(*plane_filtered, out_cloud);
    out_cloud.header.frame_id = cloud_msg->header.frame_id;
    out_cloud.header.stamp = cloud_msg->header.stamp;
    cloud_pub.publish(out_cloud);
}

bool camLidarCalib::imageHandler(const sensor_msgs::ImageConstPtr &image_msg) {
    try {
        // ros type -> cv type
        image_in = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        if (-1 == cv::findChessboardCorners(image_in, cv::Size(checkerboard_cols, checkerboard_rows), image_points, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE)) {
            return false;
        }

        cv::drawChessboardCorners(image_in, cv::Size(checkerboard_cols, checkerboard_rows), image_points, boardDetectedInCam);
        // std::cout << image_points.size() << "==" << object_points.size() << std::endl;
        if (image_points.size() == object_points.size()) {
            cv::solvePnP(object_points, image_points, projection_matrix, distCoeff, rvec, tvec, false, CV_ITERATIVE);
            projected_points.clear();
            cv::projectPoints(object_points, rvec, tvec, projection_matrix, distCoeff, projected_points, cv::noArray());
            // std::cout << projected_points.size() << std::endl;
            for (int i = 0; i < projected_points.size(); i++) {
                cv::circle(image_in, projected_points[i], 16, cv::Scalar(0, 255, 0), 10, cv::LINE_AA, 0);
            }
            cv::Rodrigues(rvec, C_R_W);
            cv::cv2eigen(C_R_W, c_R_w);
            c_t_w = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            r3 = c_R_w.block<3, 1>(0, 2);
            Nc = (r3.dot(c_t_w)) * r3;
        }
        cv::resize(image_in, image_resized, cv::Size(), 0.25, 0.25);
        cv::imshow("view", image_resized);
        cv::waitKey(10);
        return true;
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", image_msg->encoding.c_str());
        return false;
    }
}

void camLidarCalib::runSolver() {
    if (lidar_points.size() > min_points_on_plane && boardDetectedInCam) {
        if (r3.dot(r3_old) < 0.95) {  // compare roatation with pre-pcl data, need a lot of difference.
            r3_old = r3;
            all_normals.push_back(Nc);
            all_lidar_points.push_back(lidar_points);
            ROS_ASSERT(all_normals.size() == all_lidar_points.size());
            ROS_INFO_STREAM("Recording View number: " << all_normals.size());
            if (all_normals.size() >= num_views) {
                ROS_INFO_STREAM("Starting optimization...");
                init_file.open(initializations_file);
                for (int counter = 0; counter < no_of_initializations; counter++) {
                    /// Start Optimization here

                    /// Step 1: Initialization
                    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
                    Eigen::Matrix3d Rotn = transformation_matrix.block(0, 0, 3, 3);
                    Eigen::Vector3d axis_angle;
                    ceres::RotationMatrixToAngleAxis(Rotn.data(), axis_angle.data());

                    Eigen::Vector3d Translation = transformation_matrix.block(0, 3, 3, 1);

                    Eigen::Vector3d rpy_init = Rotn.eulerAngles(0, 1, 2) * 180 / M_PI;
                    Eigen::Vector3d tran_init = transformation_matrix.block(0, 3, 3, 1);

                    Eigen::VectorXd R_t(6);
                    R_t(0) = axis_angle(0);
                    R_t(1) = axis_angle(1);
                    R_t(2) = axis_angle(2);
                    R_t(3) = Translation(0);
                    R_t(4) = Translation(1);
                    R_t(5) = Translation(2);

                    /// Step2: Defining the Loss function (Can be NULL)
                    //                    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
                    //                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

                    ceres::LossFunction *loss_function = NULL;

                    /// Step 3: Form the Optimization Problem
                    ceres::Problem problem;
                    problem.AddParameterBlock(R_t.data(), 6);
                    for (int i = 0; i < all_normals.size(); i++) {
                        Eigen::Vector3d normal_i = all_normals[i];
                        std::vector<Eigen::Vector3d> lidar_points_i = all_lidar_points[i];
                        for (int j = 0; j < lidar_points_i.size(); j++) {
                            Eigen::Vector3d lidar_point = lidar_points_i[j];
                            ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CalibrationErrorTerm, 1, 6>(new CalibrationErrorTerm(lidar_point, normal_i));
                            problem.AddResidualBlock(cost_function, loss_function, R_t.data());
                        }
                    }

                    /// Step 4: Solve it
                    ceres::Solver::Options options;
                    options.max_num_iterations = 200;
                    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    //                        std::cout << summary.FullReport() << '\n';

                    /// Printing and Storing C_T_L in a file
                    ceres::AngleAxisToRotationMatrix(R_t.data(), Rotn.data());
                    Eigen::MatrixXd C_T_L(3, 4);
                    C_T_L.block(0, 0, 3, 3) = Rotn;
                    C_T_L.block(0, 3, 3, 1) = Eigen::Vector3d(R_t[3], R_t[4], R_t[5]);
                    std::cout << "RPY = " << Rotn.eulerAngles(0, 1, 2) * 180 / M_PI << std::endl;
                    std::cout << "t = " << C_T_L.block(0, 3, 3, 1) << std::endl;

                    init_file << rpy_init(0) << "," << rpy_init(1) << "," << rpy_init(2) << ","
                              << tran_init(0) << "," << tran_init(1) << "," << tran_init(2) << "\n";
                    init_file << Rotn.eulerAngles(0, 1, 2)(0) * 180 / M_PI << "," << Rotn.eulerAngles(0, 1, 2)(1) * 180 / M_PI << "," << Rotn.eulerAngles(0, 1, 2)(2) * 180 / M_PI << ","
                              << R_t[3] << "," << R_t[4] << "," << R_t[5] << "\n";

                    /// Step 5: Covariance Estimation
                    ceres::Covariance::Options options_cov;
                    ceres::Covariance covariance(options_cov);
                    std::vector<std::pair<const double *, const double *> > covariance_blocks;
                    covariance_blocks.push_back(std::make_pair(R_t.data(), R_t.data()));
                    CHECK(covariance.Compute(covariance_blocks, &problem));
                    double covariance_xx[6 * 6];
                    covariance.GetCovarianceBlock(R_t.data(), R_t.data(), covariance_xx);

                    Eigen::MatrixXd cov_mat_RotTrans(6, 6);
                    cv::Mat cov_mat_cv = cv::Mat(6, 6, CV_64F, &covariance_xx);
                    cv::cv2eigen(cov_mat_cv, cov_mat_RotTrans);

                    Eigen::MatrixXd cov_mat_TransRot(6, 6);
                    cov_mat_TransRot.block(0, 0, 3, 3) = cov_mat_RotTrans.block(3, 3, 3, 3);
                    cov_mat_TransRot.block(3, 3, 3, 3) = cov_mat_RotTrans.block(0, 0, 3, 3);
                    cov_mat_TransRot.block(0, 3, 3, 3) = cov_mat_RotTrans.block(3, 0, 3, 3);
                    cov_mat_TransRot.block(3, 0, 3, 3) = cov_mat_RotTrans.block(0, 3, 3, 3);

                    double sigma_xx = sqrt(cov_mat_TransRot(0, 0));
                    double sigma_yy = sqrt(cov_mat_TransRot(1, 1));
                    double sigma_zz = sqrt(cov_mat_TransRot(2, 2));

                    double sigma_rot_xx = sqrt(cov_mat_TransRot(3, 3));
                    double sigma_rot_yy = sqrt(cov_mat_TransRot(4, 4));
                    double sigma_rot_zz = sqrt(cov_mat_TransRot(5, 5));

                    std::cout << "sigma_xx = " << sigma_xx << "\t"
                              << "sigma_yy = " << sigma_yy << "\t"
                              << "sigma_zz = " << sigma_zz << std::endl;

                    std::cout << "sigma_rot_xx = " << sigma_rot_xx * 180 / M_PI << "\t"
                              << "sigma_rot_yy = " << sigma_rot_yy * 180 / M_PI << "\t"
                              << "sigma_rot_zz = " << sigma_rot_zz * 180 / M_PI << std::endl;

                    std::ofstream results;
                    results.open(result_str);
                    results << C_T_L;
                    results.close();

                    std::ofstream results_rpy;
                    results_rpy.open(result_rpy);
                    results_rpy << Rotn.eulerAngles(0, 1, 2) * 180 / M_PI << "\n"
                                << C_T_L.block(0, 3, 3, 1);
                    results_rpy.close();

                    // save file as json
                    writeResultToJson(C_T_L, Rotn);

                    ROS_INFO_STREAM("No of initialization: " << counter);
                }
                init_file.close();
                ros::shutdown();
            }
        } else {
            ROS_WARN_STREAM("Not enough Rotation, view not recorded");
        }
    } else {
        if (!boardDetectedInCam)
            ROS_WARN_STREAM("Checker-board not detected in Image.");
        else {
            ROS_WARN_STREAM("Checker Board Detected in Image?: " << boardDetectedInCam << "\t"
                                                                 << "No of LiDAR pts: " << lidar_points.size() << " (Check if this is less than threshold) ");
        }
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

void camLidarCalib::callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                             const sensor_msgs::ImageConstPtr &image_msg) {
    if (!imageHandler(image_msg)) return;

    // cloudHandler(cloud_msg);

    if (!handleLidarWithOpen3D(cloud_msg)) return;
    // runSolver();

    solveWithOpen3D();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "CameraLidarCalib_node");
    ros::NodeHandle nh("~");
    camLidarCalib cLC(nh);
    ros::spin();
    return 0;
}
