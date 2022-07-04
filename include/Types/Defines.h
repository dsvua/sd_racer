// #include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigen>
#include <memory>

// FAST detector parameters
#define FAST_EPSILON (13.0f)
#define FAST_MIN_ARC_LENGTH 9
// #define FAST_EPSILON (13.0f)
// #define FAST_MIN_ARC_LENGTH 12
#define FAST_SCORE SUM_OF_ABS_DIFF_ON_ARC

// NMS parameters
#define HORIZONTAL_BORDER 0
#define VERTICAL_BORDER 0
#define CELL_SIZE_WIDTH 16
#define CELL_SIZE_HEIGHT 16

#define DETECTOR_BASE_NMS_SIZE 3
// #define MINIMUM_BORDER 3
// #define FEATURE_DETECTOR_HORIZONTAL_BORDER 0
// #define FEATURE_DETECTOR_VERTICAL_BORDER 0
// #define FEATURE_DETECTOR_CELL_SIZE_WIDTH 16
// #define FEATURE_DETECTOR_CELL_SIZE_HEIGHT 16

#define FAST_GPU_USE_LOOKUP_TABLE 1
#define FAST_GPU_USE_LOOKUP_TABLE_BITBASED 1

// ds adjust floating point precision
typedef float real;

// ds existential types
typedef Eigen::Matrix<real, 3, 1> PointCoordinates;
typedef std::vector<PointCoordinates, Eigen::aligned_allocator<PointCoordinates>> PointCoordinatesVector;
typedef Eigen::Matrix<real, 3, 1> ImageCoordinates;
typedef std::vector<ImageCoordinates, Eigen::aligned_allocator<ImageCoordinates>> ImageCoordinatesVector;
typedef Eigen::Matrix<real, 3, 1> PointColorRGB;
typedef Eigen::Matrix<real, 3, 3> CameraMatrix;
typedef Eigen::Matrix<real, 3, 4> ProjectionMatrix;
typedef Eigen::Transform<real, 3, Eigen::Isometry> TransformMatrix3D;
typedef Eigen::Matrix<real, 6, 1> TransformVector3D;
typedef Eigen::Quaternion<real> Quaternion;
typedef uint32_t Identifier;
typedef uint32_t Index;
typedef uint32_t Count;
typedef std::pair<PointCoordinates, PointColorRGB> PointDrawable;

// ds generic types
typedef Eigen::Matrix<real, 2, 1> Vector2;
typedef Eigen::Matrix<real, 2, 3> Matrix2;
typedef Eigen::Matrix<real, 3, 1> Vector3;
typedef Eigen::Matrix<real, 4, 1> Vector4;
typedef Eigen::Matrix<real, 3, 3> Matrix3;
typedef Eigen::Matrix<real, 4, 4> Matrix4;
typedef Eigen::Matrix<real, 6, 6> Matrix6;
typedef Eigen::Matrix<real, 5, 1> Vector5;
typedef Eigen::Matrix<real, 6, 1> Vector6;
typedef Eigen::Matrix<real, 1, 6> Matrix1_6;
typedef Eigen::Matrix<real, 3, 6> Matrix3_6;
typedef Eigen::Matrix<real, 4, 6> Matrix4_6;
typedef Eigen::Matrix<real, 2, 6> Matrix2_6;
typedef Eigen::Matrix<real, 1, 3> Matrix1_3;
typedef Eigen::Matrix<real, 2, 3> Matrix2_3;
typedef Eigen::Matrix<real, 6, 3> Matrix6_3;
typedef Eigen::Matrix<real, 6, 4> Matrix6_4;
typedef Eigen::Matrix<real, 3, 4> Matrix3_4;
typedef Eigen::Matrix<real, 4, 3> Matrix4_3;
