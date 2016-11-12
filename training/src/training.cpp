#include <vector>
#include <Eigen/Core>
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include "pcl/io/pcd_io.h"
#include "pcl/kdtree/kdtree_flann.h"
#include "pcl/features/normal_3d.h"
#include "pcl/features/pfh.h"
#include "pcl/keypoints/sift_keypoint.h"
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <boost/thread/thread.hpp>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

using namespace pcl;
using namespace std;
typedef PointXYZRGB PointType;
typedef pcl::PointXYZ PointTypexyz;


boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis (pcl::PointCloud<PointType>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler(cloud);
  viewer->addPointCloud<PointType> (cloud, handler, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  return (viewer);
}


int main (int argc, char** argv)
{
  PointCloud<PointType>::Ptr cloud (new PointCloud<PointType>);
  PointCloud<PointType>::Ptr cloud_filtered (new PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_ransac (new PointCloud<PointType>);

  //plan segmentation - get a,b,c,d (ax + by +cz + d = 0 )
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::VectorXf coefficientsInliers;

  int tmp_coordinate_frame;
  cout << 1234;

  if (argc < 2)
  {
    PCL_ERROR ("No input file!");
    return -1;
  }
  if (io::loadPCDFile<PointType> (argv[1], *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file %s \n", argv[1]);
    return (-1);
  }
  
  PassThrough<PointType> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.5);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*cloud_filtered);

  int sumR = 0;
  int sumG = 0;
  int sumB = 0;
  int n = 0;
  for (int i = 0; i < cloud_filtered->points.size(); i++) {
    PointType &p = cloud_filtered->points[i];
    uint32_t rgb = *reinterpret_cast<int*>(&p.rgb);
    uint8_t r = (rgb >> 16) & 0x0000ff;
    uint8_t g = (rgb >> 8)  & 0x0000ff;
    uint8_t b = (rgb)       & 0x0000ff;
    sumR += r;
    sumG += g;
    sumB += b;
    n++;
  }
  int avgR = sumR / n;
  int avgG = sumG / n;
  int avgB = sumB / n;
  cout << avgR <<  " " << avgG << ' ' << avgB << '\n';

  //RANSAC algorithm
    std::vector<int> inliers;

    // created RandomSampleConsensus object and compute the appropriated model
    pcl::SampleConsensusModelSphere<PointType>::Ptr
      model_s(new pcl::SampleConsensusModelSphere<PointType> (cloud_filtered));

    pcl::SampleConsensusModelCylinder<PointType, PointNormal>::Ptr
      model_c(new pcl::SampleConsensusModelCylinder<PointType, PointNormal> (cloud_filtered));

    pcl::SampleConsensusModelPlane<PointType>::Ptr
      model_p (new pcl::SampleConsensusModelPlane<PointType> (cloud_filtered));
    if(pcl::console::find_argument (argc, argv, "-f") >= 0)
    {
      pcl::RandomSampleConsensus<PointType> ransac (model_p);
      ransac.setDistanceThreshold (.01);
      ransac.computeModel();
      ransac.getModelCoefficients(coefficientsInliers);
      ransac.getInliers(inliers);
    }
    else if (pcl::console::find_argument (argc, argv, "-sf") >= 0 )
    {
      pcl::RandomSampleConsensus<PointType> ransac (model_s);
      ransac.setDistanceThreshold (.01);
      ransac.computeModel();
      ransac.getInliers(inliers);
    }
    else if (pcl::console::find_argument (argc, argv, "-cyl") >= 0 )
    {
        pcl::NormalEstimation<PointType, pcl::Normal> ne;
        pcl::SACSegmentationFromNormals<PointType, pcl::Normal> seg;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
        pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);

        // Normal estimation
        ne.setSearchMethod (tree);
        ne.setInputCloud (cloud_filtered);
        ne.setKSearch (50);
        ne.compute (*cloud_normals);

        // Ransac
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_CYLINDER);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setNormalDistanceWeight (0.1);
        seg.setMaxIterations (10000);
        seg.setDistanceThreshold (0.05);
        seg.setRadiusLimits (0, 0.5);
        seg.setInputCloud (cloud_filtered);
        seg.setInputNormals (cloud_normals);

        // Obtain the cylinder inliers and coefficients
        seg.segment (*inliers_cylinder, *coefficients);
        inliers = inliers_cylinder->indices;
        cout << cloud_filtered->size() << '\n';
        cout << inliers_cylinder->indices.size();
    }

    // copies all inliers of the model computed to another PointCloud
    pcl::copyPointCloud<PointType>(*cloud_filtered, inliers, *cloud_ransac);

    pcl::PointCloud<PointType>::Ptr cloud_outliers(new PointCloud<PointType>());
    pcl::ExtractIndices<PointType> ei;
    ei.setInputCloud(cloud_filtered);
    pcl::PointIndices::Ptr inliers_pi (new pcl::PointIndices);
    inliers_pi->indices = inliers;
    ei.setIndices(inliers_pi);
    ei.setNegative(true);
    ei.filter(*cloud_outliers);

    // creates the visualization object and adds either our orignial cloud or all of the inliers
    // depending on the command line arguments specified.
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = simpleVis(cloud_outliers);
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    if (pcl::console::find_argument (argc, argv, "-f") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0 || pcl::console::find_argument (argc, argv, "-cyl") >= 0)
      viewer = simpleVis(cloud_ransac);
    else
      viewer = simpleVis(cloud_filtered);
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    // Plan segmentation (ax + by + cz = 0)
    copyPointCloud(*cloud_outliers, *cloud_segmented);

      pcl::ModelCoefficients::Ptr coefficientsOutliers (new pcl::ModelCoefficients);

      pcl::PointIndices::Ptr outliersSegm (new pcl::PointIndices);

      // Create the segmentation object
      pcl::SACSegmentation<pcl::PointXYZ> seg;
      // Optional
      seg.setOptimizeCoefficients (true);
      // Mandatory
      seg.setModelType (pcl::SACMODEL_PLANE);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setDistanceThreshold (0.01);

      seg.setInputCloud(cloud_segmented);
      seg.segment (*outliersSegm, *coefficientsOutliers);


      if (outliersSegm->indices.size () == 0)
      {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
        return (-1);
      }

      std::cerr << "Model OUTLIERS coefficients: " << coefficientsOutliers->values[0] << " "
                                           << coefficientsOutliers->values[1] << " "
                                           << coefficientsOutliers->values[2] << " "
                                           << coefficientsOutliers->values[3] << std::endl;

      std::cerr << "Model INLIERS coefficients: " << coefficientsInliers[0] << " "
                                          << coefficientsInliers[1] << " "
                                          << coefficientsInliers[2] << " "
                                          << coefficientsInliers[3] << std::endl;

      std::cerr << "Model ouliers: " << outliersSegm->indices.size () << std::endl;
      std::cerr << "Model inliers: " << inliers_pi->indices.size () << std::endl;

      //compute scalar product of two normals (outliers and inliers) to determine if are perpendicular
      if((coefficientsInliers[0]*coefficientsOutliers->values[0] +
              coefficientsInliers[1]*coefficientsOutliers->values[1] +
                  coefficientsInliers[2]*coefficientsOutliers->values[2]) > (-0.1)
              && (coefficientsInliers[0]*coefficientsOutliers->values[0] +
                      coefficientsInliers[1]*coefficientsOutliers->values[1] +
                          coefficientsInliers[2]*coefficientsOutliers->values[2]) < (0.1))

        {

             cout<<"perpedicular"<<endl;
        }
        else
        {

             cout<<"Not perpedicular"<<endl;
        }

  pcl::PointCloud<PointXYZ>::Ptr cloud_final(new PointCloud<PointXYZ>());
  copyPointCloud(*cloud_segmented, outliersSegm->indices, *cloud_final);

  io::savePCDFileASCII ("../data/result_pcd.pcd", *cloud_final);

  return (0);
}
