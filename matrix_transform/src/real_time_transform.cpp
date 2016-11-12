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
#include <pcl/segmentation/extract_clusters.h>

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

PointCloud<PointType>::Ptr cloud(new PointCloud<PointType>);
PointCloud<PointType>::Ptr cloud_object(new PointCloud<PointType>);
PointCloud<PointType>::Ptr cloud_filtered(new PointCloud<PointType>); //passthrough filter
PointCloud<PointType>::Ptr cloud_filtered2(new PointCloud<PointType>); //voxelgrid filter
pcl::PointCloud<PointType>::Ptr cloud_ransac(new PointCloud<PointType>);
pcl::PointCloud<PointXYZ>::Ptr cloud_final(new PointCloud<PointXYZ>());
PointCloud<PointType>::Ptr cloud_without_ground(new PointCloud<PointType>);

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

hsv rgb2hsv(rgb in) {
    hsv out;
    double min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min < in.b ? min : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max > in.b ? max : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001) {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0
        // s = 0, v is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if (in.r >= max)                    // > is bogus, just keeps compilor happy
        out.h = (in.g - in.b) / delta;        // between yellow & magenta
    else if (in.g >= max)
        out.h = 2.0 + (in.b - in.r) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + (in.r - in.g) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if (out.h < 0.0)
        out.h += 360.0;

    return out;
}

const char* figuriArray[] = { "plane", "sphere", "cylinder", "other" };
const char* colorArray[] = { "red", "green", "blue" };

void passthroughFilter() //passthrough filter
{
    PassThrough<PointType> pass;
    pass.setInputCloud(cloud);
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
    cout << color_hsv.h << " " << color_hsv.s << " " << color_hsv.v << '\n';

    return color_hsv;
}

void planDetection() {
    pcl::SampleConsensusModelPlane<PointType>::Ptr model_p(
            new pcl::SampleConsensusModelPlane<PointType>(cloud_object));

    pcl::RandomSampleConsensus<PointType> ransac(model_p);
    ransac.setDistanceThreshold(.01);
    ransac.computeModel();
    ransac.getModelCoefficients(coefficientsInliers);
    ransac.getInliers(inliers_plan);

}

void sphereDetection() {
    pcl::SampleConsensusModelSphere<PointType>::Ptr model_s(
            new pcl::SampleConsensusModelSphere<PointType>(cloud_object));

    pcl::RandomSampleConsensus<PointType> ransac(model_s);
    ransac.setDistanceThreshold(.01);
    model_s->setRadiusLimits(0.0, 0.5);
    ransac.computeModel();
    ransac.getInliers(inliers_sphere);

}

void cylinderDetection() {
    pcl::SampleConsensusModelCylinder<PointType, PointNormal>::Ptr model_c(
            new pcl::SampleConsensusModelCylinder<PointType, PointNormal>(
                    cloud_object));

    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointType, pcl::Normal> seg;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
            new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::search::KdTree<PointType>::Ptr tree(
            new pcl::search::KdTree<PointType>());

    // Normal estimation
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_object);
    ne.setKSearch(10);
    ne.compute(*cloud_normals);

    // Ransac
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.05);
    seg.setRadiusLimits(0, 0.5);
    seg.setInputCloud(cloud_object);
    seg.setInputNormals(cloud_normals);

    // Obtain the cylinder inliers and coefficients
    seg.segment(*inliers_cylinder, *coefficients);
}

void objectsDetection(vector<int> inliers) {
    // copies all inliers of the model computed to another PointCloud
    pcl::copyPointCloud<PointType>(*cloud_object, inliers, *cloud_ransac);
}

void voxelGrid()
{
    // Create the filtering object
      pcl::VoxelGrid<PointType> sor;
      sor.setInputCloud(cloud_filtered);
      sor.setLeafSize (0.01f, 0.01f, 0.01f);
      sor.filter(*cloud_filtered2);

      std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
           << " data points (" << pcl::getFieldsList (*cloud_filtered) << ").";
}

void getOutliers(PointCloud<PointType>::Ptr cloud_in, vector<int> inliers, PointCloud<PointType>::Ptr cloud_out) {
    pcl::ExtractIndices<PointType> ei;
    ei.setInputCloud(cloud_in);
    inliers_pi->indices = inliers;
    ei.setIndices(inliers_pi);
    ei.setNegative(true);
    ei.filter(*cloud_out);
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

    /*
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
    */

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
        //cout << "Object detected" << std::endl;
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

string detectShape() {

    //colorDetection();

    /*if (pcl::console::find_argument(argc, argv, "-f") >= 0) {
     planDetection();
     } else if (pcl::console::find_argument(argc, argv, "-sf") >= 0) {
     sphereDetection();
     } else if (pcl::console::find_argument(argc, argv, "-cyl") >= 0) {
     cylinderDetection();
     }*/

    const double ratio = 60.0 / 100;
    const int minPoints = (int) (cloud_object->points.size() * ratio);
    sphereDetection();
    if (inliers_sphere.size() >= minPoints) {
        //sfera
        return string("sfera");
    } else {
        cylinderDetection();
        if (inliers_cylinder->indices.size() >= minPoints) {
            //cilindru
            return string("cilindru");
        } else {
            planDetection();
            if (inliers_plan.size() >= minPoints) {
                //plan
                return string("plan");
            } else {
                //object (box)
                getOutliers(cloud_object, inliers_plan, cloud_outliers);
                if (planSegmentation()) {
                    return string("obiect");
                } else {
                    return string("necunoscut");
                }
            }
        }
    }
    cout << '\n';
}

// input: cloud_filtered
// output: cloud_without_ground
void findGround()
{
    pcl::SampleConsensusModelPlane<PointType>::Ptr model_p(
                new pcl::SampleConsensusModelPlane<PointType>(cloud_filtered2));

    pcl::RandomSampleConsensus<PointType> ransac(model_p);
    ransac.setDistanceThreshold(.01);
    ransac.computeModel();
    ransac.getModelCoefficients(coefficientsInliers);
    ransac.getInliers(inliers_plan);
    getOutliers(cloud_filtered2, inliers_plan, cloud_without_ground);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D viewer"));
// input: cloud_without_ground
void euclidianClustering() {
    pcl::search::KdTree<PointType>::Ptr tree(
            new pcl::search::KdTree<PointType>);
    tree->setInputCloud(cloud_without_ground);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance(0.02); // 2cm
    ec.setMinClusterSize(1000);
    //ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_without_ground);
    ec.extract(cluster_indices);

    int j = 0;
    viewer->removeAllPointClouds(0);
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler(
            cloud_without_ground);
    viewer->addPointCloud<PointType>(cloud_without_ground, handler, "sample cloud");
    viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    cout << cluster_indices.size() << "clusters\n";
    for (std::vector<pcl::PointIndices>::const_iterator it =
            cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        PointXYZ pointMin(99999.0f, 99999.0f, 99999.0f);
        PointXYZ pointMax(-99999.0f, -99999.0f, -99999.0f);
        float sx = 0;
        float sy = 0;
        float sz = 0;
        cloud_object->clear();
        for (std::vector<int>::const_iterator pit = it->indices.begin();
                pit != it->indices.end(); ++pit) {
            PointType &p = cloud_without_ground->points[*pit];
            cloud_object->points.push_back(p);
            sx += p.x;
            sy += p.y;
            sz += p.z;

            if (p.x < pointMin.x) {
                pointMin.x = p.x;
            }
            if (p.y < pointMin.y) {
                pointMin.y = p.y;
            }
            if (p.z < pointMin.z) {
                pointMin.z = p.z;
            }

            if (p.x > pointMax.x) {
                pointMax.x = p.x;
            }
            if (p.y > pointMax.y) {
                pointMax.y = p.y;
            }
            if (p.z > pointMax.z) {
                pointMax.z = p.z;
            }
        }
        int numberOfPoints = it->indices.size();
        PointXYZ centroid(sx / numberOfPoints, sy / numberOfPoints, sz / numberOfPoints);
        cloud_object->width = cloud_object->points.size();
        cloud_object->height = 1;
        cloud_object->is_dense = true;

        j++;
        cout << "Object " << j << ':' << cloud_object->points.size() << "points;\n";
        string shape = detectShape();
        stringstream ss;
        ss << "box" << j;
        viewer->addCube(
                pointMin.x, pointMax.x,
                pointMin.y, pointMax.y,
                pointMin.z, pointMax.z,
                1.0, 1.0, 1.0,
                ss.str(), 0
        );
        ss << "text";
        viewer->addText3D(shape.c_str(), pointMin, 0.1, 0.0, 1.0, 0.0, ss.str(), 0);
    }
    //while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    //    boost::this_thread::sleep(boost::posix_time::microseconds(100000));

    //}
    viewer->removeAllShapes(0);
}

void cloud_cb(
        const boost::shared_ptr<const sensor_msgs::PointCloud2>& cloud_msg) {
    pcl::PCLPointCloud2 cloud_pcl;
    pcl_conversions::toPCL(*cloud_msg, cloud_pcl); //first from sensor msgs pointcloud2 to pcl pointcloud2
    pcl::fromPCLPointCloud2(cloud_pcl, *cloud); //from pcl pointcloud2 to pcl pointxyz in order to make to computations below

    passthroughFilter();
    voxelGrid();
    findGround();
    euclidianClustering();
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

