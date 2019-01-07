#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

int
main (int argc, char** argv)
{
  // Load input file into a PointCloud<T> with an appropriate type
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2 cloud_blob;
  pcl::io::loadPCDFile ("/home/zach/Work/pcl-pcl-1.8.0/test/bun0.pcd", cloud_blob);
  pcl::fromPCLPointCloud2 (cloud_blob, *cloud);
  //* the data should be available in cloud

  // Normal estimation*
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud);
  n.setInputCloud (cloud);
  n.setSearchMethod (tree);
  n.setKSearch (20);
  n.compute (*normals);
  //* normals should not contain the point normals + surface curvatures

  // Concatenate the XYZ and normal fields*
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
  //* cloud_with_normals = cloud + normals

  // Create search tree*
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud (cloud_with_normals);

  // Initialize objects
  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
  pcl::PolygonMesh triangles;

  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (0.025);

  // Set typical values for the parameters
  gp3.setMu (2.5);
  gp3.setMaximumNearestNeighbors (100);
  gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
  gp3.setMinimumAngle(M_PI/18); // 10 degrees
  gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
  gp3.setNormalConsistency(false);

  // Get result
  gp3.setInputCloud (cloud_with_normals);
  gp3.setSearchMethod (tree2);
  gp3.reconstruct (triangles);

  // Additional vertex information
  std::vector<int> parts = gp3.getPartIDs();
  std::vector<int> states = gp3.getPointStates();

  // Finish
  return (0);
}

// #include <getopt.h>
// #include <cstdlib>
// #include <iostream>
// #include <iomanip>
//
// #include <pcl/point_types.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/surface/gp3.h>
//
// #include <pcl/io/vtk_io.h>
//
// int
// main (int argc, char** argv)
// {
//
//   std::string usage =
//   "\n"
//   "Usage: WriteAsciiToMSH <options>\n"
//   "\n"
//   "This program creates a triangular mesh from an MSH file.\n"
//   "\n"
//   "\tAvailable options:\n"
//   "\t -h, --help               (shows this message and exits)\n"
//   "\t -f, --file               (specify the input triangles.txt file)\n"
//   "\n";
//
//   static struct option longOptions[] = {
//       {"help", no_argument, 0, 'h'},
//       {"file", required_argument, 0, 'f'}
//   };
//
//   static const char *optString = "ha:b:n:m:s:";
//
//   std::string inFile = "";
//
//   while(1)
//   {
//       char optId = getopt_long(argc, argv, optString, longOptions, NULL);
//       if(optId == -1){
//           break;
//       }
//       switch(optId) {
//       case('h'): // help
//           std::cout<<usage<<std::endl;
//       return 1;
//       case('f'):
//           inFile = std::string(optarg);
//       break;
//       default: // unrecognized option
//       return 1;
//       }
//   }
//
//   std::string suffix = inFile.substr(inFile.find_last_of("."),std::string::npos);
//
//   struct stat fileInfo;
//   bool exists;
//   int fileStat;
//
//   // Attempt to get the file attributes
//   fileStat = stat(inFile.c_str(),&fileInfo);
//   if(fileStat == 0)
//   exists = true;
//   else
//   exists = false;
//
//   if (!exists) {
//   std::cout << "Error: file \"" << inFile <<"\" cannot be read." << std::endl;
//   return 1;
//   }
//
//   if (suffix.compare(".pcd") != 0) {
//       std::cout<<"Error: unkown file extension \""<<suffix<<"\"" << std::endl;
//       return 1;
//   }
//
//   // The original version of this code (http://pointclouds.org/documentation/tutorials/greedy_projection.php) assumes that only coordinates are available, without normals
//
//   // Load input file into a PointCloud<T> with an appropriate type
//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//   pcl::PCLPointCloud2 cloud_blob;
//   pcl::io::loadPCDFile (inFile, cloud_blob);
//   pcl::fromPCLPointCloud2 (cloud_blob, *cloud);
//   //* the data should be available in cloud
//
//   // Normal estimation*
//   pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
//   pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
//   pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//   tree->setInputCloud (cloud);
//   n.setInputCloud (cloud);
//   n.setSearchMethod (tree);
//   n.setKSearch (20);
//   n.compute (*normals);
//   //* normals should not contain the point normals + surface curvatures
//
//   // Concatenate the XYZ and normal fields*
//   pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
//   pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
//   //* cloud_with_normals = cloud + normals
//
//   //Simplified version using pre-determined normals
//
//   // pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
//   // pcl::PCLPointCloud2 cloud_blob;
//   // pcl::io::loadPCDFile (inFile, cloud_blob);
//   // pcl::fromPCLPointCloud2 (cloud_blob, *cloud_with_normals);
//
//   //Everything else is the same whether normals are given or estimated
//
//   // Create search tree*
//   pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
//   tree2->setInputCloud (cloud_with_normals);
//
//   // Initialize objects
//   pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
//   pcl::PolygonMesh triangles;
//
//   // Set the maximum distance between connected points (maximum edge length)
//   gp3.setSearchRadius (0.025);
//
//   // Set typical values for the parameters
//   gp3.setMu (2.5);
//   gp3.setMaximumNearestNeighbors (100);
//   gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
//   gp3.setMinimumAngle(M_PI/18); // 10 degrees
//   gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
//   gp3.setNormalConsistency(false);
//
//   // Get result
//   gp3.setInputCloud (cloud_with_normals);
//   gp3.setSearchMethod (tree2);
//   gp3.reconstruct (triangles); //crash occurs here
//
//   // Additional vertex information
//   std::vector<int> parts = gp3.getPartIDs();
//   std::vector<int> states = gp3.getPointStates();
//
//   pcl::io::saveVTKFile ("mesh.vtk", triangles);
//
//   // Finish
//   return (0);
// }
