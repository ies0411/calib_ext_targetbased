#include "cam_lidar_calib.h"

bool camLidarCalib::cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
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
    pass_x.setFilterLimits(x_min_, x_max_);
    pass_x.filter(*cloud_filtered_x);

    pcl::PassThrough<pcl::PointXYZ> pass_y;
    pass_y.setInputCloud(cloud_filtered_x);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(y_min_, y_max_);
    pass_y.filter(*cloud_filtered_y);

    pcl::PassThrough<pcl::PointXYZ> pass_z;
    pass_z.setInputCloud(cloud_filtered_y);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(z_min_, z_max_);
    pass_z.filter(*cloud_filtered_z);
    // ransac_threshold = 0.01;
    /// Plane Segmentation
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
        new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_z));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
    ransac.setDistanceThreshold(ransac_threshold_);
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
    lidar_points_.clear();
    for (size_t i = 0; i < plane_filtered->points.size(); i++) {
        double X = plane_filtered->points[i].x;
        double Y = plane_filtered->points[i].y;
        double Z = plane_filtered->points[i].z;
        lidar_points_.push_back(Eigen::Vector3d(X, Y, Z));
    }
    //        ROS_INFO_STREAM("No of planar_pts: " << lidar_points.size());
    std::cout << "No of planar_pts: " << plane_filtered->points.size() << std::endl;
    if (plane_filtered->points.size() < min_points_on_plane_) return false;

    // sensor_msgs::PointCloud2 out_cloud;
    // open3d_ros::open3dToRos(*plane_cloud_, out_cloud, "o3d_frame");
    // cloud_pub_.publish(out_cloud);
    // if (plane_cloud_->points_.size() < min_points_on_plane_) return false;
    sensor_msgs::PointCloud2 out_cloud;
    pcl::toROSMsg(*plane_filtered, out_cloud);
    // out_cloud.header.frame_id = cloud_msg->header.frame_id;
    // out_cloud.header.stamp = cloud_msg->header.stamp;
    cloud_pub_.publish(out_cloud);
    return true;
}

bool camLidarCalib::handleLidarWithOpen3D(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    std::shared_ptr<open3d::geometry::PointCloud> pcd = std::make_shared<open3d::geometry::PointCloud>();

    open3d_ros::rosToOpen3d(cloud_msg, *pcd);

    std::shared_ptr<open3d::geometry::PointCloud> filtered_pcd = std::make_shared<open3d::geometry::PointCloud>();
    filtered_pcd->points_.clear();
    for (size_t i = 0; i < pcd->points_.size(); i++) {
        const Eigen::Vector3d tmp_point = pcd->points_[i];
        if (tmp_point[X_] > x_min_ && tmp_point[X_] < x_max_ &&
            tmp_point[Y_] > y_min_ && tmp_point[Y_] < y_max_ &&
            tmp_point[Z_] > z_min_ && tmp_point[Z_] < z_max_) {
            filtered_pcd->points_.push_back(tmp_point);
        }
    }
    auto [plane_model, plane_points] = filtered_pcd->SegmentPlane(0.001, 5, 500);

    // auto [plane_model, plane_points] = filtered_pcd->SegmentPlane(0.05, 5, 200);

    plane_cloud_ = filtered_pcd->SelectByIndex(plane_points);

    sensor_msgs::PointCloud2 out_cloud;
    open3d_ros::open3dToRos(*plane_cloud_, out_cloud, "o3d_frame");
    cloud_pub_.publish(out_cloud);
    if (plane_cloud_->points_.size() < min_points_on_plane_) return false;
    return true;
}
bool camLidarCalib::solveWithOpen3D() {
    if (filtering_) {
        if (r3_.dot(r3_old_) > 0.95) {
            ROS_WARN("same position data..");
            data_size_--;
            return false;
        }
    }
    r3_old_ = r3_;
    all_normals_.push_back(Nc_);
    all_points_.push_back(plane_cloud_->points_);
    // all_points_.push_back(lidar_points_);
    ROS_INFO("num of view : %d", all_normals_.size());
    if (file_type_ == "file") {
        if ((all_normals_.size() < num_views_) && (all_normals_.size() < data_size_ + 1)) return false;
    } else {
        if (all_normals_.size() < num_views_) return false;
    }
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
bool camLidarCalib::solveWithPCL() {
    if (filtering_) {
        if (r3_.dot(r3_old_) > 0.95) {
            ROS_WARN("same position data..");
            data_size_--;
            std::cout << data_size_ << std::endl;
            if ((all_normals_.size() < data_size_ - 1)) return false;
        }
    }
    r3_old_ = r3_;
    all_normals_.push_back(Nc_);
    all_lidar_points_.push_back(lidar_points_);

    /////

    ROS_INFO("num of view : %d", all_normals_.size());
    std::cout << data_size_ << std::endl;
    if (file_type_ == "file") {
        if ((all_normals_.size() < num_views_) && (all_normals_.size() < data_size_ - 1)) return false;
    } else {
        if (all_normals_.size() < num_views_) return false;
    }
    ROS_INFO_STREAM("Starting optimization...");
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    rotation_matrix << 0, -1, 0, 0, 0, -1, 1, 0, 0;
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
        std::vector<Eigen::Vector3d> lidar_point_i = all_lidar_points_[i];
        for (int j = 0; j < lidar_point_i.size(); j++) {
            Eigen::Vector3d lidar_point = lidar_point_i[j];
            ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CalibrationErrorTerm, 1, 6>(new CalibrationErrorTerm(lidar_point, normal_i));
            problem.AddResidualBlock(cost_function, loss_function, opt_param.data());
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 500;
    // options.preconditioner_type = ceres::SCHUR_JACOBI;

    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.linear_solver_type = ceres::DENSE_SCHUR;

    options.minimizer_progress_to_stdout = true;
    // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

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
    ROS_INFO("setting..");
    priv_nh_.param<std::string>("file_type", file_type_, "file");
    priv_nh_.param<double>("dx", dx_, 0.075);
    priv_nh_.param<double>("dy", dy_, 0.075);
    priv_nh_.param<int>("checkerboard_rows", checkerboard_rows_, 6);
    priv_nh_.param<int>("checkerboard_cols", checkerboard_cols_, 9);

    priv_nh_.param<int>("min_points_on_plane", min_points_on_plane_, 450);
    priv_nh_.param<int>("num_views", num_views_, 10);
    priv_nh_.param<double>("publish_rate", publish_rate_, 10);

    priv_nh_.param<std::string>("json_result_file", result_json_, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/result_json.json"));
    priv_nh_.param<std::string>("load_img_path", load_img_path_, std::string("/home/catkin_ws/src/"));
    priv_nh_.param<std::string>("load_pcd_path", load_pcd_path_, std::string("/home/catkin_ws/src/"));
    priv_nh_.param<int>("num_data", num_data_, 10);
    priv_nh_.param<std::string>("result_file", result_str_, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/CTL.txt"));
    priv_nh_.param<std::string>("cam_config_file_path", cam_config_file_path_, std::string("/home/catkin_ws/src/cam_lidar_calib/config/intrinsic.yaml"));
    priv_nh_.param<double>("x_min", x_min_, 0);
    priv_nh_.param<double>("x_max", x_max_, 6);
    priv_nh_.param<double>("y_min", y_min_, -1.25);
    priv_nh_.param<double>("y_max", y_max_, 1.25);
    priv_nh_.param<double>("z_min", z_min_, -0.5);
    priv_nh_.param<double>("z_max", z_max_, 2.0);
    priv_nh_.param<double>("ransac_threshold_", ransac_threshold_, 0.01);
    priv_nh_.param<bool>("show_debug", show_debug_, false);
    priv_nh_.param<bool>("filtering", filtering_, false);

    priv_nh_.param<std::string>("img_type", img_type_, std::string("png"));
    // priv_nh_.param<std::string>("plane_library", plane_library_, std::string("open3d"));

    priv_nh_.param<std::string>("camera_in_topic", camera_in_topic_, std::string("/pylon_camera_node/image_raw"));
    priv_nh_.param<std::string>("lidar_in_topic", lidar_in_topic_, std::string("/velodyne_points"));
}

void camLidarCalib::readCameraParams(std::string cam_config_file_path_, int &image_height_, int &image_width_, cv::Mat &D, cv::Mat &K) {
    cv::FileStorage fs_cam_config(cam_config_file_path_, cv::FileStorage::READ);
    if (!fs_cam_config.isOpened())
        std::cerr << "Error: Wrong path: " << cam_config_file_path_ << std::endl;
    fs_cam_config["image_height"] >> image_height_;
    fs_cam_config["image_width"] >> image_width_;
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
        if (show_debug_) {
            cv::resize(image_in_, image_resized_, cv::Size(), 0.25, 0.25);
            cv::imshow("view", image_resized_);
            cv::waitKey(10);
        }
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
    if (!imageHandler(image_msg)) {
        data_size_--;
        if ((all_normals_.size() < data_size_ - 1)) return;
    }

    // if (!handleLidarWithOpen3D(cloud_msg)) {
    //     data_size_--;
    //     if ((all_normals_.size() < data_size_ - 1)) return;
    // }

    if (!cloudHandler(cloud_msg)) {
        data_size_--;
        if ((all_normals_.size() < data_size_ - 1)) return;
    }
    // TODO : make param for switch open3d or pcl
    // solveWithOpen3D();
    solveWithPCL();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "CameraLidarCalib_node");
    ros::NodeHandle nh_("~");
    camLidarCalib cLC(nh_);

    if (cLC.getFiletype() == "rosbag") {
        ros::spin();
    } else if (cLC.getFiletype() == "file") {
        ROS_INFO("file to rosbag");
        // std::vector<cv::String> images;
        std::vector<cv::Mat> images;
        std::vector<open3d::geometry::PointCloud> pcds;
        // cv::glob(cLC.getImagefilePath(), images);
        // int size = images.size();
        // ROS_INFO("number of images : %d", size);
        // Test
        // std::string file_path=cLC.getImagefilePath()+%d;
        // cv::Mat test = cv::imread(file_path);
        // TODO :jpg or png
        std::cout << "data loading.." << std::endl;
        for (int i = 0; i < cLC.returnNumOfData(); i++) {
            cv::Mat img = cv::imread(cLC.getImagefilePath() + std::to_string(i) + "." + cLC.img_type_, cv::IMREAD_COLOR);
            images.push_back(img);
            auto pcd = std::make_shared<open3d::geometry::PointCloud>();
            open3d::io::ReadPointCloud(cLC.getPcdfilePath() + std::to_string(i) + ".pcd", *pcd);
            pcds.push_back(*pcd);
        }

        /***********/
        // for (const auto &p : std::experimental::filesystem::recursive_directory_iterator(cLC.getPcdfilePath())) {
        //     auto pcd = std::make_shared<open3d::geometry::PointCloud>();
        //     open3d::io::ReadPointCloud(p.path(), *pcd);
        //     pcds.push_back(*pcd);
        // }
        cLC.data_size_ = images.size();
        ROS_INFO("size : %d", cLC.data_size_);
        if (images.size() != pcds.size()) {
            ROS_WARN("pcd's size is not equal to img's size");
            return -1;
        }
        ROS_INFO("complete setup..");
        u_int idx = 0;
        ros::Rate rate(cLC.publish_rate_);
        while (ros::ok()) {
            // cv::Mat frame = cv::imread(images[idx]);
            cv::Mat frame = images[idx];
            auto pcd = pcds[idx++];

            cLC.publishImgAndPCD(cLC.covertCVtoROS(frame), cLC.converPCDtoROS(pcd));
            // if (idx == cLC.returnNumOfData()) {
            //     ROS_WARN("need more various img");
            //     return -1;
            // }

            ros::spinOnce();
            rate.sleep();
        }
    }

    return 0;
}
