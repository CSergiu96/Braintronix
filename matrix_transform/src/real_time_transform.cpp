#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include "pcl/io/pcd_io.h"
#include "pcl/kdtree/kdtree_flann.h"
#include "pcl/features/normal_3d.h"

ros::Publisher pub;

using namespace pcl;
using namespace std;

//convert RGB in HSV to determine colors
typedef struct {
    double r;       // percent
    double g;       // percent
    double b;       // percent
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // percent
    double v;       // percent
} hsv;


//------------------------------
//----Global variables----------
//------------------------------
typedef PointXYZRGB PointType;

PointCloud<PointType>::Ptr cloud_object(new PointCloud<PointType>);
PointCloud<PointType>::Ptr cloud_filtered(new PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr cloud_ransac(new PointCloud<PointType>);
pcl::PointCloud<PointXYZ>::Ptr cloud_final(new PointCloud<PointXYZ>());

//plan segmentation - get a,b,c,d (ax + by +cz + d = 0 )
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented(
        new pcl::PointCloud<pcl::PointXYZ>);
Eigen::VectorXf coefficientsInliers;

//RANSAC algorithm
//std::vector<int> inliers;
std::vector<int> inliers_sphere;
std::vector<int> inliers_plan;
pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);


pcl::PointCloud<PointType>::Ptr cloud_outliers(new PointCloud<PointType>());

pcl::PointIndices::Ptr inliers_pi(new pcl::PointIndices);


hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0
            // s = 0, v is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}

const char* figuriArray[] = {"plane", "sphere", "cylinder", "other"};
const char* colorArray[] = {"red", "green", "blue"};

void objectFilter() //passthrough filter
{
    PassThrough<PointType> pass;
    pass.setInputCloud(cloud_object);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 1.5);
    //pass.setFilterLimitsNegative (true);
    pass.filter(*cloud_filtered);
}

hsv colorDetection() {
    int sumR = 0, sumG = 0, sumB = 0, n = 0;

    for (int i = 0; i < cloud_filtered->points.size(); i++) {
        PointType &p = cloud_filtered->points[i];
        uint32_t rgb = *reinterpret_cast<int*>(&p.rgb);
        uint8_t r = (rgb >> 16) & 0x0000ff;
        uint8_t g = (rgb >> 8) & 0x0000ff;
        uint8_t b = (rgb) & 0x0000ff;
        sumR += r;
        sumG += g;
        sumB += b;
        n++;
    }
    int avgR = sumR / n;
    int avgG = sumG / n;
    int avgB = sumB / n;

    rgb color;
    color.r = avgR / 255.0;
    color.g = avgG / 255.0;
    color.b = avgB / 255.0;
    hsv color_hsv = rgb2hsv(color);
    cout << color_hsv.h << " " << color_hsv.s  << " " << color_hsv.v  << '\n';

    return color_hsv;
}

void planDetection() {
    pcl::SampleConsensusModelPlane<PointType>::Ptr model_p(
            new pcl::SampleConsensusModelPlane<PointType>(cloud_filtered));

    pcl::RandomSampleConsensus<PointType> ransac(model_p);
    ransac.setDistanceThreshold(.01);
    ransac.computeModel();
    ransac.getModelCoefficients(coefficientsInliers);
    ransac.getInliers(inliers_plan);

    cout << "Plan detected" << std::endl;

}

void sphereDetection() {
    pcl::SampleConsensusModelSphere<PointType>::Ptr model_s(
            new pcl::SampleConsensusModelSphere<PointType>(cloud_filtered));

    pcl::RandomSampleConsensus<PointType> ransac(model_s);
    ransac.setDistanceThreshold(.01);
    model_s->setRadiusLimits(0.0, 0.3);
    ransac.computeModel();
    ransac.getInliers(inliers_sphere);
    cout << "Sphere detected" << std::endl;

}

void cylinderDetection() {
    pcl::SampleConsensusModelCylinder<PointType, PointNormal>::Ptr model_c(
            new pcl::SampleConsensusModelCylinder<PointType, PointNormal>(
                    cloud_filtered));

    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointType, pcl::Normal> seg;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
            new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::search::KdTree<PointType>::Ptr tree(
            new pcl::search::KdTree<PointType>());

    // Normal estimation
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_filtered);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);

    // Ransac
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.05);
    seg.setRadiusLimits(0, 0.5);
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);

    // Obtain the cylinder inliers and coefficients
    seg.segment(*inliers_cylinder, *coefficients);

    cout << "Cylinder detected" << std::endl;
}

void objectsDetection(vector<int> inliers) {
    // copies all inliers of the model computed to another PointCloud
    pcl::copyPointCloud<PointType>(*cloud_filtered, inliers, *cloud_ransac);
}

void getOutliers(vector<int> inliers) {
    pcl::ExtractIndices<PointType> ei;
    ei.setInputCloud(cloud_filtered);
    inliers_pi->indices = inliers;
    ei.setIndices(inliers_pi);
    ei.setNegative(true);
    ei.filter(*cloud_outliers);
}

bool planSegmentation() {
    // Plan segmentation (ax + by + cz = 0)
    copyPointCloud(*cloud_outliers, *cloud_segmented);

    pcl::ModelCoefficients::Ptr coefficientsOutliers(
            new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr outliersSegm(new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    seg.setInputCloud(cloud_segmented);
    seg.segment(*outliersSegm, *coefficientsOutliers);

    if (outliersSegm->indices.size() == 0) {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return false;
    }

    std::cerr << "Model OUTLIERS coefficients: "
            << coefficientsOutliers->values[0] << " "
            << coefficientsOutliers->values[1] << " "
            << coefficientsOutliers->values[2] << " "
            << coefficientsOutliers->values[3] << std::endl;

    std::cerr << "Model INLIERS coefficients: " << coefficientsInliers[0] << " "
            << coefficientsInliers[1] << " " << coefficientsInliers[2] << " "
            << coefficientsInliers[3] << '\n';

    std::cerr << "Model ouliers: " << outliersSegm->indices.size() << std::endl;
    std::cerr << "Model inliers: " << inliers_pi->indices.size() << std::endl;

    copyPointCloud(*cloud_segmented, outliersSegm->indices, *cloud_final);
    //compute scalar product of two normals (outliers and inliers) to determine if are perpendicular
    if ((coefficientsInliers[0] * coefficientsOutliers->values[0]
            + coefficientsInliers[1] * coefficientsOutliers->values[1]
            + coefficientsInliers[2] * coefficientsOutliers->values[2]) > (-0.1)
            && (coefficientsInliers[0] * coefficientsOutliers->values[0]
                    + coefficientsInliers[1] * coefficientsOutliers->values[1]
                    + coefficientsInliers[2] * coefficientsOutliers->values[2])
                    < (0.1))

                    {
        cout << "Object detected" << std::endl;
        return true;
    }
    return false;
}

typedef PointXYZRGB PointType;

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(
        pcl::PointCloud<PointType>::ConstPtr cloud) {
    // -----Open 3D viewer and add point cloud-----
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler(
            cloud);
    viewer->addPointCloud<PointType>(cloud, handler, "sample cloud");
    viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    //viewer->addCoordinateSystem (1.0, "global");
    viewer->initCameraParameters();
    return (viewer);
}

void cloud_cb(
        const boost::shared_ptr<const sensor_msgs::PointCloud2>& cloud_msg) {
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Container for original & filtered data
    pcl::PCLPointCloud2 cloud; // = new pcl::PCLPointCloud2;
    //pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    //pcl::PCLPointCloud2 cloud_filtered;
    pcl::PointCloud<PointType>::Ptr cloud_pcl(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_filtered(
            new pcl::PointCloud<PointType>());
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, cloud); //first from sensor msgs pointcloud2 to pcl pointcloud2
    pcl::fromPCLPointCloud2(cloud, *cloud_pcl); //from pcl pointcloud2 to pcl pointxyz in order to make to computations below
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

int main(int argc, char** argv) {
    // Initialize ROS
    ros::init(argc, argv, "real_time_transform");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe("/camera/depth_registered/points", 1,
            cloud_cb);

    // Create a ROS publisher for the output point cloud
    pub = nh.advertise<sensor_msgs::PointCloud2>("output", 1);

    // Spin
    ros::spin();
}

