#ifndef __CUSTOM_OPT_H__
#define __CUSTOM_OPT_H__

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "ros/ros.h"

enum POINT_DIRECTION {
    X_,
    Y_,
    Z_,
};

struct pointsPair {
    Eigen::Vector2d prj;
    Eigen::Vector3d pc;
};

//  cv::Mat projection_matrix_;
//     cv::Mat distCoeff_;
class customOptimize {
   private:
    Eigen::Vector2d img_ptr_;
    Eigen::Vector3d pc_ptr_;
    Eigen::Matrix3d projection_matrix_;
    Eigen::Vector4d distortion_;
    // const Eigen::Vector3d normal_to_plane_;

   public:
    customOptimize(const pointsPair &points_pair, const cv::Mat projection_matrix, const cv::Mat distCoeff) {
        img_ptr_ = points_pair.prj;
        pc_ptr_ = points_pair.pc;
        cv::cv2eigen(projection_matrix, projection_matrix_);
        cv::cv2eigen(distCoeff, distortion_);
        // std::cout << "projection_matrix_ : " << std::endl
        //           << projection_matrix_ << std::endl;
        // std::cout << "distortion_ : " << std::endl
        //           << distortion_ << std::endl;
    }

    template <typename T>
    bool operator()(const T *const R_t, T *residuals) const {
        Eigen::Matrix<T, 3, 3> innerT = projection_matrix_.cast<T>();
        Eigen::Matrix<T, 4, 1> distorT = distortion_.cast<T>();

        T l_pt_L[3] = {T(pc_ptr_(0)), T(pc_ptr_(1)), T(pc_ptr_(2))};
        T l_pt_C[3];
        ceres::AngleAxisRotatePoint(R_t, l_pt_L, l_pt_C);
        l_pt_C[0] += R_t[3];
        l_pt_C[1] += R_t[4];
        l_pt_C[2] += R_t[5];

        T xo = l_pt_C[0] / l_pt_C[2];
        T yo = l_pt_C[0] / l_pt_C[2];

        const T &fx = innerT.coeffRef(0, 0);
        const T &cx = innerT.coeffRef(0, 2);
        const T &fy = innerT.coeffRef(1, 1);
        const T &cy = innerT.coeffRef(1, 2);

        T r2 = xo * xo + yo * yo;

        T r4 = r2 * r2;
        T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;

        T xd = xo * distortion + T(2) * distorT[2] * xo * yo + distorT[3] * (r2 + T(2) * xo * xo);
        T yd = yo * distortion + T(2) * distorT[3] * xo * yo + distorT[2] * (r2 + T(2) * yo * yo);

        T ud = fx * xd + cx;
        T vd = fy * yd + cy;

        // std::cout << ud << "," << vd << std::endl;
        residuals[0] = ud - T(img_ptr_[X_]);
        residuals[1] = vd - T(img_ptr_[Y_]);

        return true;
    }
};
#endif
