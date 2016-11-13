#include <ros/ros.h>
#include <std_msgs/String.h>
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

Eigen::VectorXf ground_coefficients;

pcl::PointCloud<PointType>::Ptr cloud_outliers(new PointCloud<PointType>());

pcl::PointIndices::Ptr inliers_pi(new pcl::PointIndices);



enum ShapeType {
    BOX, CYLINDER, SPHERE, PERSON, UNKNOWN, OUR_OBJECT, NOT_OUR_OBJECT
};

struct Shape
{
    ShapeType type;
    hsv color;
    vector<double> coefficients;
    PointXYZ center;
    PointXYZ pointMin;
    PointXYZ pointMax;
};

struct ObjectFilter {
    ShapeType shapeType;
    int index;
};

ObjectFilter command;
bool commandReceived = false;
bool shouldDetect = true;

double hDist(double h1, double h2) {
    double minH = (h1 < h2 ? h1 : h2);
    double maxH = (h1 >= h2 ? h1 : h2);

    double d1 = maxH - minH;
    double d2 = minH + 360.0 - maxH;
    return (d1 < d2 ? d1 : d2);
}

string getColor(hsv color) {
    if (color.s < 0.1 || color.v < 0.1) {
        if (color.v > 0.5) {
            return "white";
        }
        else {
            return "black";
        }
    }
    double distRed = hDist(color.h, 0);
    double distGreen = hDist(color.h, 100);
    double distBlue = hDist(color.h, 230);

    if (distRed < distGreen && distRed < distBlue) {
        return "red";
    }
    if (distGreen < distBlue) {
        return "green";
    }
    return "blue";
}

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

// input: cloud_object
hsv colorDetection() {
    int sumR = 0, sumG = 0, sumB = 0, n = 0;

    for (int i = 0; i < cloud_object->points.size(); i++) {
        PointType &p = cloud_object->points[i];
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

    return color_hsv;
}

void planDetection() {
    pcl::SampleConsensusModelPlane<PointType>::Ptr model_p(
            new pcl::SampleConsensusModelPlane<PointType>(cloud_object));

    pcl::RandomSampleConsensus<PointType> ransac(model_p);
    ransac.setDistanceThreshold(.02);
    ransac.computeModel();
    ransac.getModelCoefficients(coefficientsInliers);
    ransac.getInliers(inliers_plan);

}

void sphereDetection() {
    pcl::SampleConsensusModelSphere<PointType>::Ptr model_s(
            new pcl::SampleConsensusModelSphere<PointType>(cloud_object));

    pcl::RandomSampleConsensus<PointType> ransac(model_s);
    ransac.setDistanceThreshold(.02);
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
    seg.setDistanceThreshold(0.02);

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

string shapeTypeToString(ShapeType type)
{
    switch (type)
    {
    case CYLINDER:
        return string("cylinder");
    case SPHERE:
        return string("sphere");
    case BOX:
        return string("box");
    case PERSON:
        return string("person");
    case UNKNOWN:
        return string("unknown");
    case OUR_OBJECT:
        return string("our_object");
    case NOT_OUR_OBJECT:
        return string("not_our_object");
    default:
        return string("PROBLEM IN shapeTypeToString");
    }
}

Shape detectShape() {
    Shape res;

    const double ratio = 60.0 / 100;
    const int minPoints = (int) (cloud_object->points.size() * ratio);
    const int min_remaining_points_for_object = 200;
    int bestPoints = 0;
    bool nocheck = false;
    string bestPointsString;
    sphereDetection();
    if (inliers_sphere.size() > bestPoints) {
        bestPoints = inliers_sphere.size();
        res.type = SPHERE;
    }

    cylinderDetection();
    if (inliers_cylinder->indices.size() > bestPoints) {
        bestPoints = inliers_cylinder->indices.size();
        res.type = CYLINDER;
    }

    planDetection();
    if (inliers_plan.size() > bestPoints) {
        bestPoints = inliers_plan.size();
        res.type = BOX;
    }

    //object (box)
    getOutliers(cloud_object, inliers_plan, cloud_outliers);
    if (cloud_outliers->points.size() >= min_remaining_points_for_object) {
        if (planSegmentation()) {
            res.type = BOX;
            nocheck = true;
        }
    }
    if (!nocheck && bestPoints < minPoints) {
        // TODO: unknown, not person
        res.type = PERSON;
    }

    return res;
}

// input: cloud_filtered
// output: cloud_without_ground, ground_coefficients
void findGround()
{
    pcl::SampleConsensusModelPlane<PointType>::Ptr model_p(
                new pcl::SampleConsensusModelPlane<PointType>(cloud_filtered2));

    pcl::RandomSampleConsensus<PointType> ransac(model_p);
    ransac.setDistanceThreshold(.01);
    ransac.computeModel();
    ransac.getModelCoefficients(ground_coefficients);
    ransac.getInliers(inliers_plan);
    double groundPlaneYNormal = abs(ground_coefficients[1]);
    if (groundPlaneYNormal > 0.9) { // it is vertical
        //cout << "Ground detected!\n";
        getOutliers(cloud_filtered2, inliers_plan, cloud_without_ground);
    }
    else {
        //cout << "Ground not detected!\n";
        copyPointCloud(*cloud_filtered2, *cloud_without_ground);
    }
}

bool cmpByX(const Shape& s1, const Shape& s2) {
    return s1.center.x < s2.center.x;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D viewer"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_raw (new pcl::visualization::PCLVisualizer("Raw"));
// input: cloud_without_ground
PointXYZ last_location;

double distance_xyz(PointXYZ p1, PointXYZ p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

double MAX_DISTANCE = 0.3;
ShapeType checkOurObject(PointXYZ center) {
    if (distance_xyz(last_location, center) < MAX_DISTANCE) {
        return OUR_OBJECT;
    }
    else {
        return NOT_OUR_OBJECT;
    }
}

void euclidianClustering() {
    pcl::search::KdTree<PointType>::Ptr tree(
            new pcl::search::KdTree<PointType>);
    tree->setInputCloud(cloud_without_ground);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance(0.03); // 2cm
    ec.setMinClusterSize(1000);
    //ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_without_ground);
    ec.extract(cluster_indices);

    int j = 0;
    vector<Shape> shapes[10];
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

        Shape shape;
        if (shouldDetect) {
            shape = detectShape();
        } else {
            shape.type = checkOurObject(centroid);
            if (shape.type == OUR_OBJECT) {
                last_location = centroid;
            }
        }
        shape.color = colorDetection();
        shape.pointMin = pointMin;
        shape.pointMax = pointMax;
        shape.center = centroid;

        shapes[shape.type].push_back(shape);
    }

    // DRAWING:
    viewer->removeAllPointClouds(0);
    viewer->removeAllShapes(0);
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler(
            cloud_without_ground);
    viewer->addCoordinateSystem(0.3, 0);
    viewer->addPointCloud<PointType>(cloud_without_ground, handler, "sample cloud");
    viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    for (int shapeType = 0; shapeType < 9; shapeType++) {
    sort(shapes[shapeType].begin(), shapes[shapeType].end(), cmpByX);
    for (int i = 0; i < shapes[shapeType].size(); i++) {
        if (commandReceived && shouldDetect) {
            if (command.shapeType == shapeType && command.index == i) {
                cout << "GASITU-L-AM";
                last_location = shapes[shapeType][i].center;
                shouldDetect = false;
            }
        }
        j++;
        stringstream ss;
        ss << "box" << j;
        rgb box_color;
        if (shapes[shapeType][i].type == OUR_OBJECT) {
            box_color.r = 1.0;
            box_color.g = 0.0;
            box_color.b = 0.0;
        } else {
            box_color.r = 1.0;
            box_color.g = 1.0;
            box_color.b = 1.0;
        }
        viewer->addCube(
                shapes[shapeType][i].pointMin.x, shapes[shapeType][i].pointMax.x,
                shapes[shapeType][i].pointMin.y, shapes[shapeType][i].pointMax.y,
                shapes[shapeType][i].pointMin.z, shapes[shapeType][i].pointMax.z,
                box_color.r, box_color.g, box_color.b,
                ss.str(), 0
        );
        ss << "text";
        stringstream displayStr;
        displayStr << shapeTypeToString(shapes[shapeType][i].type);
        if (shapes[shapeType].size() > 1) {
            displayStr << "_" << i + 1;
        }
        displayStr << "_" << getColor(shapes[shapeType][i].color);
        viewer->addText3D(displayStr.str(), shapes[shapeType][i].pointMin, 0.1, 0.0, 1.0, 0.0, ss.str(), 0);
        ss << "center";
        viewer->addSphere(shapes[shapeType][i].center, 0.03, 1.0, 0.0, 0.0, ss.str(), 0);
    }
    }
    viewer->spinOnce(100);
}

void cloud_cb(
        const boost::shared_ptr<const sensor_msgs::PointCloud2>& cloud_msg) {
    clock_t start_time = clock();
    pcl::PCLPointCloud2 cloud_pcl;
    pcl_conversions::toPCL(*cloud_msg, cloud_pcl); //first from sensor msgs pointcloud2 to pcl pointcloud2
    pcl::fromPCLPointCloud2(cloud_pcl, *cloud); //from pcl pointcloud2 to pcl pointxyz in order to make to computations below

    // Drawing:
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_raw(
            cloud);
    viewer_raw->removeAllPointClouds(0);
    viewer_raw->addPointCloud(cloud, handler_raw, "cloud_raw", 0);
    viewer_raw->spinOnce(100);

    passthroughFilter();
    voxelGrid();
    findGround();
    euclidianClustering();
    clock_t end_time = clock();
    cout << "Detection took " << (double) (end_time - start_time) / CLOCKS_PER_SEC << "seconds\n";
    ros::spinOnce();
}

void command_cb(const std_msgs::StringConstPtr& commandStr)
{
    string c = string(commandStr->data.c_str());
    vector<string> tokens;
    istringstream iss(c);
    copy(istream_iterator<string>(iss),
         istream_iterator<string>(),
         back_inserter(tokens));
    command.shapeType = UNKNOWN;
    for (int i = 0; i < tokens.size(); i++) {
        cout << tokens[i] << '\n';
        if (tokens[i] == "box") {
            command.shapeType = BOX;
        } else if (tokens[i] == "cylinder") {
            command.shapeType = CYLINDER;
        } else if (tokens[i] == "person") {
            // TODO: person
            command.shapeType = PERSON;
        } else if (tokens[i] == "sphere") {
            command.shapeType = SPHERE;
        }

        if (tokens[i] == "first") {
            command.index = 0;
        } else if (tokens[i] == "second") {
            command.index = 1;
        } else if (tokens[i] == "third") {
            command.index = 2;
        } else if (tokens[i] == "fourth") {
            command.index = 3;
        }
    }
    if (command.shapeType == UNKNOWN) {
        return;
    }

    commandReceived = true;
    // TODO: process the string
    cout << command.shapeType << '\n';
}

int main(int argc, char** argv) {
    // Initialize ROS
    ros::init(argc, argv, "real_time_transform");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe("/camera/depth_registered/points", 1,
            cloud_cb);

    // Service creation
    ros::Subscriber sub2 = nh.subscribe("/robot/command", 1, command_cb);

    // Create a ROS publisher for the output point cloud
    pub = nh.advertise<sensor_msgs::PointCloud2>("output", 1);

    // Spin
    ros::spin();
}

