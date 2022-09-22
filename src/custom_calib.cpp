#include "custom_calib.hpp"

void customCalib::createMatchingPoints(std::shared_ptr<open3d::geometry::PointCloud> pcd) {
    int idx = 0;
    for (int i = 0; i < pcd->points_.size(); i++) {
        pointsPair tmp_pair;
        tmp_pair.pc = pcd->points_[i];
        std::vector<cv::Point2d> imagepoints;
        std::vector<cv::Point3d> pointcloud;
        pointcloud.push_back(cv::Point3d(pcd->points_[i][0], pcd->points_[i][1], pcd->points_[i][2]));
        cv::projectPoints(pointcloud, rvec_, tvec_, projection_matrix_, distCoeff_, imagepoints, cv::noArray());
        if ((imagepoints[0].x > 0) && (imagepoints[0].x < image_width_) && (imagepoints[0].y > 0) && (imagepoints[0].y < image_height_)) {
            tmp_pair.prj[X_] = imagepoints[0].x;
            tmp_pair.prj[Y_] = imagepoints[0].y;
            points_pair_.push_back(tmp_pair);
        }
    }
}
void customCalib::makeNoise() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist;
    for (auto &p : points_pair_) {
        p.prj[X_] += std::round(dist(gen));
        p.prj[Y_] += std::round(dist(gen));
    }
}

void customCalib::handleLidarWithOpen3D(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
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

    auto [plane_model, plane_points] = filtered_pcd->SegmentPlane(0.005, 5, 100);
    plane_cloud_ = filtered_pcd->SelectByIndex(plane_points);
    createMatchingPoints(pcd);
    makeNoise();
    solveOpt();
    // sensor_msgs::PointCloud2 out_cloud;
    // open3d_ros::open3dToRos(*plane_cloud_, out_cloud, "o3d_frame");
    // cloud_pub_.publish(out_cloud);
    // if (plane_cloud_->points_.size() < min_points_on_plane_) return;
}

void customCalib::findInitialValue(Eigen::Vector6d &opt_param) {
    // double pi = 3;
    double axis_x = 0., axis_y = 0., axis_z = 0.;
    double trl_x = 0., trl_y = 0., trl_z = 0.;
    Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();
    Eigen::Vector3d axis_angle;
    Eigen::Vector3d best_angle, best_trl;
    double best_cost = 100.0;
    double center = 1.2092;
    double res = 0.1;
    for (double i = center - 0.5; i <= center + 0.5; i = i + res) {
        for (double j = -center - 0.5; j <= -center + 0.5; j = j + res) {
            for (double k = center - 0.5; k <= center + 0.5; k = k + res) {
                axis_angle << axis_x + i, axis_y + j, axis_z + k;
                opt_param(0) = axis_angle(0);
                opt_param(1) = axis_angle(1);
                opt_param(2) = axis_angle(2);
                opt_param(3) = translation_vector(0);
                opt_param(4) = translation_vector(1);
                opt_param(5) = translation_vector(2);

                ceres::LossFunction *loss_function = nullptr;
                ceres::Problem problem;
                problem.AddParameterBlock(opt_param.data(), opt_param.size());

                for (int cnt = 0; cnt < 30; cnt++) {
                    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<customOptimize, 1, 6>(new customOptimize(points_pair_[cnt], projection_matrix_, distCoeff_));
                    problem.AddResidualBlock(cost_function, loss_function, opt_param.data());
                }

                double total_cost = 0.0;
                ceres::Solver::Options options;
                options.max_num_iterations = 20;
                // options.preconditioner_type = ceres::SCHUR_JACOBI;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.minimizer_progress_to_stdout = false;
                // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                problem.Evaluate(ceres::Problem::EvaluateOptions(), &total_cost, nullptr, nullptr, nullptr);
                if (best_cost > total_cost) {
                    best_cost = total_cost;
                    best_angle[X_] = axis_angle[X_];
                    best_angle[Y_] = axis_angle[Y_];
                    best_angle[Z_] = axis_angle[Z_];
                }
            }
        }
    }
    std::cout << "best_cost: " << std::endl
              << best_cost << std::endl;
    std::cout << "best angle : " << std::endl
              << best_angle << std::endl;

    best_cost = 100.0;
    for (double i = -0.5; i <= 0.5; i = i + 0.2) {
        for (double j = -0.5; j <= 0.5; j = j + 0.2) {
            for (double k = -0.5; k <= 0.5; k = k + 0.2) {
                translation_vector << trl_x + i, trl_y + j, trl_z + k;
                opt_param(0) = best_angle(0);
                opt_param(1) = best_angle(1);
                opt_param(2) = best_angle(2);
                opt_param(3) = translation_vector(0);
                opt_param(4) = translation_vector(1);
                opt_param(5) = translation_vector(2);

                ceres::LossFunction *loss_function = nullptr;
                ceres::Problem problem;
                problem.AddParameterBlock(opt_param.data(), opt_param.size());

                for (int i = 0; i < 30; i++) {
                    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<customOptimize, 1, 6>(new customOptimize(points_pair_[i], projection_matrix_, distCoeff_));
                    problem.AddResidualBlock(cost_function, loss_function, opt_param.data());
                }

                double total_cost = 0.0;
                ceres::Solver::Options options;
                options.max_num_iterations = 20;
                // options.preconditioner_type = ceres::SCHUR_JACOBI;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.minimizer_progress_to_stdout = false;
                // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                problem.Evaluate(ceres::Problem::EvaluateOptions(), &total_cost, nullptr, nullptr, nullptr);
                if (best_cost > total_cost) {
                    best_cost = total_cost;
                    best_trl[X_] = translation_vector[X_];
                    best_trl[Y_] = translation_vector[Y_];
                    best_trl[Z_] = translation_vector[Z_];
                }
                // std::cout << "total : " << total_cost << std::endl;
            }
        }
    }
    std::cout << "best trans : " << std::endl
              << best_trl << std::endl;
    opt_param(0) = best_angle(0);
    opt_param(1) = best_angle(1);
    opt_param(2) = best_angle(2);
    opt_param(3) = best_trl(0);
    opt_param(4) = best_trl(1);
    opt_param(5) = best_trl(2);
}

bool customCalib::solveOpt() {
    // matching pairs

    // for (int i = 0; i < 40; i++) {
    //     // p.prj
    //     // std::cout << "check" << std::endl;
    //     cv::Point2f img_pt;
    //     img_pt.x = points_pair_[i].prj[X_];
    //     img_pt.y = points_pair_[i].prj[Y_];

    //     cv::Point3f pcd;
    //     pcd.x = points_pair_[i].pc[X_];
    //     pcd.y = points_pair_[i].pc[Y_];
    //     pcd.z = points_pair_[i].pc[Z_];
    //     // std::cout << img_pt << std::endl;
    //     // std::cout << pcd << std::endl;

    //     objectPoints_.push_back(pcd);
    //     imagePoints_.push_back(img_pt);
    // }

    for (int i = 0; i < 80; i++) {
        double img_x = 0, img_y = 0, pcd_x = 0, pcd_y = 0, pcd_z = 0;
        std::vector<std::pair<double, double>> img_vec;
        std::vector<std::pair<double, double>> pcd_vec;

        for (int j = 0; j < 5; j++) {
            img_x += points_pair_[i].prj[X_];
            img_y += points_pair_[i].prj[Y_];
            img_vec.push_back(std::make_pair(img_x, img_y));
            pcd_x += points_pair_[i].pc[X_];
            pcd_y += points_pair_[i].pc[Y_];
            pcd_z += points_pair_[i].pc[Z_];
        }

        cv::Point2f img_pt;
        img_pt.x = img_x / 5;
        img_pt.y = img_y / 5;

        cv::Point3f pcd;
        pcd.x = pcd_x / 5;
        pcd.y = pcd_y / 5;
        pcd.z = pcd_z / 5;

        objectPoints_.push_back(pcd);
        imagePoints_.push_back(img_pt);
    }

    cv::Mat rvec, tvec;  // rotation & translation vectors
    // cv::solvePnP(objectPoints_, imagePoints_, projection_matrix_, distCoeff_, rvec, tvec, false, cv::SOLVEPNP_EPNP);

    cv::solvePnPRansac(objectPoints_, imagePoints_, projection_matrix_, distCoeff_, rvec, tvec, false, cv::SOLVEPNP_UPNP);
    // cv::solvePnPRefineLM(objectPoints_, imagePoints_, projection_matrix_, distCoeff_, rvec, tvec);
    Eigen::Matrix3d rotation_matrix;
    cv::Mat R;

    cv::Rodrigues(rvec, R);

    cv::Mat R_inv = R.inv();

    cv::Mat P = -R_inv * tvec;

    double *p = (double *)P.data;

    // camera position
    std::cout << R << std::endl
              << tvec << std::endl;
    // printf("x=%lf, y=%lf, z=%lf", p[0], p[1], p[2]);
    // //  = Eigen::Matrix3d::Identity();
    // Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();
    // Eigen::Vector3d axis_angle;
    // // axis_angle << 1.25975, -1.15591, 1.27355;
    // //  1.2092
    // // -1.2092
    // //  1.2092
    // //  1.25975, -1.15591, 1.27355;

    // // ceres::RotationMatrixToAngleAxis(rotation_matrix.data(), axis_angle.data());  //  convert to Rodrigus type
    // // std::cout << axis_angle << std::endl;
    // Eigen::Vector6d opt_param;
    // findInitialValue(opt_param);

    // // opt_param(0) = axis_angle(0);
    // // opt_param(1) = axis_angle(1);
    // // opt_param(2) = axis_angle(2);
    // // opt_param(3) = translation_vector(0);
    // // opt_param(4) = translation_vector(1);
    // // opt_param(5) = translation_vector(2);

    // std::cout << "opt" << opt_param << std::endl;
    // for (int i = 0; i < 1; i++) {
    //     // ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    //     ceres::LossFunction *loss_function = nullptr;
    //     ceres::Problem problem;
    //     problem.AddParameterBlock(opt_param.data(), opt_param.size());
    //     std::cout << "size : " << (int)points_pair_.size() << std::endl;
    //     for (int i = 0; i < 30; i++) {
    //         ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<customOptimize, 1, 6>(new customOptimize(points_pair_[i], projection_matrix_, distCoeff_));
    //         problem.AddResidualBlock(cost_function, loss_function, opt_param.data());
    //     }
    //     double total_cost = 0.0;

    //     ceres::Solver::Options options;
    //     options.max_num_iterations = 500;
    //     options.preconditioner_type = ceres::JACOBI;
    //     // options.preconditioner_type = ceres::SUBSET;
    //     options.linear_solver_type = ceres::DENSE_SCHUR;
    //     // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    //     // options.linear_solver_type = ceres::SPARSE_SCHUR;
    //     options.minimizer_progress_to_stdout = false;
    //     options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, &problem, &summary);
    //     std::cout << summary.BriefReport() << std::endl;
    //     // calibra.rotation_matrix_ = m_q.toRotationMatrix();
    //     // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    //     axis_angle(0) = opt_param(0);
    //     axis_angle(1) = opt_param(1);
    //     axis_angle(2) = opt_param(2);
    //     ceres::AngleAxisToRotationMatrix(axis_angle.data(), rotation_matrix.data());
    //     translation_vector(0) = opt_param(3);
    //     translation_vector(1) = opt_param(4);
    //     translation_vector(2) = opt_param(5);
    //     std::cout << "rotation\n"
    //               << rotation_matrix
    //               << "\ntranslation\n"
    //               << translation_vector << std::endl;
    // }
    // std::ofstream results;
    // results.open(result_str_);
    // results << "rotation\n"
    //         << rotation_matrix
    //         << "\ntranslation\n"
    //         << translation_vector;
    // results.close();
    // ROS_INFO_STREAM("complete..");
    ros::shutdown();
}
void customCalib::getParamFunc(ros::NodeHandle &priv_nh_) {
    ROS_INFO("setting..");
    priv_nh_.param<std::string>("file_type", file_type_, "rosbag");
    // priv_nh_.param<int>("min_points_on_plane", min_points_on_plane_, 450);

    priv_nh_.param<std::string>("json_result_file", result_json_, std::string("/home/catkin_ws/src/cam_lidar_calib/debug_data/result_json.json"));
    priv_nh_.param<std::string>("load_pcd_path", load_pcd_path_, std::string("/home/catkin_ws/src/"));
    priv_nh_.param<int>("num_data", num_data_, 10);
    priv_nh_.param<std::string>("result_file", result_str_, std::string(""));
    priv_nh_.param<std::string>("cam_config_file_path", cam_config_file_path_, std::string("/home/catkin_ws/src/cam_lidar_calib/config/intrinsic.yaml"));
    priv_nh_.param<double>("x_min", x_min_, 0);
    priv_nh_.param<double>("x_max", x_max_, 6);
    priv_nh_.param<double>("y_min", y_min_, -1.25);
    priv_nh_.param<double>("y_max", y_max_, 1.25);
    priv_nh_.param<double>("z_min", z_min_, -0.5);
    priv_nh_.param<double>("z_max", z_max_, 2.0);
    priv_nh_.param<double>("ransac_threshold_", ransac_threshold_, 0.01);

    priv_nh_.param<std::string>("lidar_in_topic", lidar_in_topic_, std::string("/velodyne_points"));
}

void customCalib::readCameraParams(std::string cam_config_file_path_, int &image_height_, int &image_width_, cv::Mat &D, cv::Mat &K) {
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

void customCalib::writeResultToJson(Eigen::MatrixXd &C_T_L, Eigen::Matrix3d &Rotn) {
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

int main(int argc, char **argv) {
    ros::init(argc, argv, "CameraLidarCalib_node");
    ros::NodeHandle nh_("~");
    customCalib cLC(nh_);
    if (cLC.getFiletype() == "rosbag") {
        ros::spin();
    } else if (cLC.getFiletype() == "file") {
        ROS_INFO("file to rosbag");
        std::vector<open3d::geometry::PointCloud> pcds;
        for (int i = 0; i < cLC.returnNumOfData(); i++) {
            auto pcd = std::make_shared<open3d::geometry::PointCloud>();
            open3d::io::ReadPointCloud(cLC.getPcdfilePath() + std::to_string(i) + ".pcd", *pcd);
            pcds.push_back(*pcd);
        }
        u_int idx = 0;
        ros::Rate rate(1);
        while (ros::ok()) {
            auto pcd = pcds[idx++];
            cLC.publishPCD(cLC.converPCDtoROS(pcd));
            ros::spinOnce();
            rate.sleep();
        }
    }
    return 0;
}
