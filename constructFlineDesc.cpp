
/* 目标:对输入的model,scene计算.匹配关系,并估计位姿
 * 下采样->计算关键点(iss)->计算特征(fpfh)
 * 可视化点云以及关键点,并可视化其特征直方图
 */
#include <pcl/registration/ia_ransac.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <iostream>
#include <pcl/keypoints/iss_3d.h>
#include <cstdlib>
#include <pcl/visualization/pcl_plotter.h>// 直方图的可视化 方法2
#include <pcl/registration/sample_consensus_prerejective.h>   // pose estimate
#include <pcl/pcl_macros.h>
#include <fstream>  
#include <string>  
#include <vector> 

using namespace std;
using pcl::NormalEstimation;
using pcl::search::KdTree;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;


typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;


// 估计法线
// input:cloud
// output:normals
void
est_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::Normal>::Ptr normals)
{

	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	norm_est.setNumberOfThreads(4);   //手动设置线程数
	norm_est.setKSearch(10);         //设置k邻域搜索阈值为10个点
	norm_est.setInputCloud(cloud_in);   //设置输入模型点云
	norm_est.compute(*normals);//计算点云法线

}

// 计算fpfh特征
// input: keypoints ,cloud , normals
// output: FPFH descriptors
void
compute_fpfh(pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors)
{
	clock_t start = clock();
	// FPFH estimation object.

	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	fpfh.setInputCloud(keypoints);  // 计算keypoints处的特征
	fpfh.setInputNormals(normals);   // cloud的法线
	fpfh.setSearchSurface(cloud); // 计算的平面是cloud 而不是keypoints
	fpfh.setSearchMethod(kdtree);
	// Search radius, to look for neighbors. Note: the value given here has to be
	// larger than the radius used to estimate the normals.
	fpfh.setRadiusSearch(0.02);
	cout << "fpfh start... " << endl;
	fpfh.compute(*descriptors);

	for (int i = 0; i < descriptors->points.size(); i++) {
		pcl::FPFHSignature33 desc = descriptors->points[i];
		cout << desc << endl;
	}

	clock_t end = clock();
	cout << "Time fpfh: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
	cout << "Get fpfh: " << descriptors->points.size() << endl;

}


/*
// fpfh match
// input: modelDescriptors,sceneDescriptors
// output: pcl::CorrespondencesPtr
void
find_match(pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors, pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors, pcl::CorrespondencesPtr model_scene_corrs)
{
	clock_t start = clock();

	pcl::KdTreeFLANN<pcl::FPFHSignature33> matching;
	matching.setInputCloud(model_descriptors);

	for (size_t i = 0; i < scene_descriptors->size(); ++i)
	{
		std::vector<int> neighbors(1);
		std::vector<float> squaredDistances(1);
		// Ignore NaNs.
		if (pcl_isfinite(scene_descriptors->at(i).histogram[0]))
		{
			// Find the nearest neighbor (in descriptor space)...
			int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1, neighbors, squaredDistances);
			// ...and add a new correspondence if the distance is less than a threshold
			// (SHOT distances are between 0 and 1, other descriptors use different metrics).
			if (neighborCount == 1 && squaredDistances[0] < 0.1f)
			{
				pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i), squaredDistances[0]);
				model_scene_corrs->push_back(correspondence);
				cout << "( " << correspondence.index_query << "," << correspondence.index_match << " )" << endl;
			}
		}
	}

	std::cout << "Found " << model_scene_corrs->size() << " correspondences." << std::endl;
	clock_t end = clock();
	cout << "Time match: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
	cout << "-----------------------------" << endl;
}
*/
/*
// 位姿估计
// input: model,scene,model_descriptors,scene_descriptors
// output: R,t
void
estimationPose(pcl::PointCloud<pcl::PointXYZ>::Ptr model, pcl::PointCloud<pcl::PointXYZ>::Ptr scene,
	pcl::PointCloud<pcl::PFHSignature125>::Ptr model_descriptors, pcl::PointCloud<pcl::PFHSignature125>::Ptr scene_descriptors,
	pcl::PointCloud<pcl::PointXYZ>::Ptr alignedModel)
{
	// Object for pose estimation.
	pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::PFHSignature125> pose;
	pose.setInputSource(model);
	pose.setInputTarget(scene);
	pose.setSourceFeatures(model_descriptors);
	pose.setTargetFeatures(scene_descriptors);
	// Instead of matching a descriptor with its nearest neighbor, choose randomly between
	// the N closest ones, making it more robust to outliers, but increasing time.
	pose.setCorrespondenceRandomness(5);   // 匹配最近的5个描述子
	// Set the fraction (0-1) of inlier points required for accepting a transformation.
	// At least this number of points will need to be aligned to accept a pose.
	pose.setInlierFraction(0.01f);    //内点的数量
	// Set the number of samples to use during each iteration (minimum for 6 DoF is 3).
	pose.setNumberOfSamples(3);     // 估计6DOF需要3个点
	// Set the similarity threshold (0-1 between edge lengths of the polygons. The
	// closer to 1, the more strict the rejector will be, probably discarding acceptable poses.
	pose.setSimilarityThreshold(0.2f);
	// Set the maximum distance threshold between two correspondent points in source and target.
	// If the distance is larger, the points will be ignored in the alignment process.
	pose.setMaxCorrespondenceDistance(1.0f);  // 允许的最大距离

	pose.setMaximumIterations(50000);   // 迭代次数

	pose.align(*alignedModel);

	if (pose.hasConverged())
	{
		Eigen::Matrix4f transformation = pose.getFinalTransformation();
		Eigen::Matrix3f rotation = transformation.block<3, 3>(0, 0);
		Eigen::Vector3f translation = transformation.block<3, 1>(0, 3);

		std::cout << "Transformation matrix:" << std::endl << std::endl;
		printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
		printf("\t\tR = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
		printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
		std::cout << std::endl;
		printf("\t\tt = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
	}
	else std::cout << "Did not converge." << std::endl;
}

//点云可视化
// 显示model+scene以及他们的keypoints
void
visualize_pcd(PointCloud::Ptr model, PointCloud::Ptr model_keypoints, PointCloud::Ptr scene, PointCloud::Ptr scene_keypoints)
{
	pcl::visualization::PCLVisualizer viewer("registration Viewer");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_color(model, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_color(scene, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_keypoint_color(model_keypoints, 0, 0, 255);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_keypoint_color(scene_keypoints, 0, 0, 255);
	viewer.setBackgroundColor(255, 255, 255);
	viewer.addPointCloud(model, model_color, "model");
	viewer.addPointCloud(scene, scene_color, "scene");
	viewer.addPointCloud(model_keypoints, model_keypoint_color, "model_keypoints");
	viewer.addPointCloud(scene_keypoints, scene_keypoint_color, "scene_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "model_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scene_keypoints");


	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

// 可视化直方图
void
visualize_Histogram(pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_feature, pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_feature)
{
	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*model_feature, 33); //设置的很坐标长度，该值越大，则显示的越细致
	plotter.addFeatureHistogram(*scene_feature, 33); //设置的很坐标长度，该值越大，则显示的越细致
	plotter.plot();
}
*/
int main(int argc, char** argv)
{
	//PointCloud::Ptr aligned_model(new PointCloud);    // 变换之后的点云
	//pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());  // model-scene的匹配关系


	pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());  // fpfh特征
	//pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());


	PointCloud::Ptr model_keypoint(new PointCloud);                 // 特征线上关键点
	//PointCloud::Ptr scene_keypoint(new PointCloud);


	PointCloud::Ptr cloud_src_model(new PointCloud);    //model点云  读入
	//PointCloud::Ptr cloud_src_scene(new PointCloud);    //scene点云
	PointCloud::Ptr model(new PointCloud);    //model点云  降采样后,一切计算是以这个为基础的
	//PointCloud::Ptr scene(new PointCloud);    //scene点云
	pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);  // 法向量
	//pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);


	//加载点云
	
	if (pcl::io::loadPCDFile("bun00.pcd", *cloud_src_model) == -1)
	{
		std::cout << "Cloud reading failed." << std::endl;
		return (-1);
	}

	pcl::io::loadPCDFile("bun00.pcd", *cloud_src_model);
	//pcl::io::loadPCDFile("bun045.pcd", *cloud_src_scene);
	cout << "/////////////////////////////////////////////////" << endl;
	cout << "原始model点云数量：" << cloud_src_model->size() << endl;
	//cout << "原始scene点云数量：" << cloud_src_scene->size() << endl;


	model = cloud_src_model;
	//scene = cloud_src_scene;

	// 计算点云的法线
	est_normals(model, model_normals);
	//est_normals(scene, scene_normals);

	// 读入特征线上的点
	// 加载txt数据
	typedef struct tagPOINT_3D
	{
		double x;  //mm world coordinate x  
		double y;  //mm world coordinate y  
		double z;  //mm world coordinate z  
		double r;
	}POINT_WORLD;
	int number_Txt;
	FILE *fp_txt;
	tagPOINT_3D TxtPoint;
	vector<tagPOINT_3D> m_vTxtPoints;
	fp_txt = fopen("line0004-1.txt", "r");
	if (fp_txt)
	{
		while (fscanf(fp_txt, "%lf %lf %lf", &TxtPoint.x, &TxtPoint.y, &TxtPoint.z) != EOF)
		{
			m_vTxtPoints.push_back(TxtPoint);
		}
	}
	else
		cout << "txt数据加载失败！" << endl;
	free(fp_txt);

	number_Txt = m_vTxtPoints.size();
	pcl::PointCloud<pcl::PointXYZ>::Ptr fline(new pcl::PointCloud<pcl::PointXYZ>);
	// Fill in the cloud data  
	fline->width = number_Txt;
	fline->height = 1;
	fline->is_dense = false;
	fline->points.resize(fline->width * fline->height);
	for (size_t i = 0; i < fline->points.size(); ++i)
	{
		fline->points[i].x = m_vTxtPoints[i].x;
		fline->points[i].y = m_vTxtPoints[i].y;
		fline->points[i].z = m_vTxtPoints[i].z;
	}
	
	std::cerr << fline->points.size() << std::endl;

	for (size_t i = 0; i < fline->points.size(); ++i)
	{
		cout << fline->points[i].x << " " << fline->points[i].y << " "<< fline->points[i].z << endl;
	}

	// 根据关键点计算特征
	
    compute_fpfh(fline,model,model_normals,model_descriptors);   // fpfh特征
   // compute_fpfh(scene_keypoint,scene,scene_normals,scene_descriptors);


	cout << "output fpfh " << endl;
	// 特征描述的输出
	for (int i = 0; i < model_descriptors->points.size(); i++) {
		pcl::FPFHSignature33 descriptor = model_descriptors->points[i];
		cout << descriptor << endl;
	}

	// maxpooling


	// match  得到对应关系:model_scene_corrs
//    find_match(model_descriptors,scene_descriptors,model_scene_corrs);        // fpfh

	// pose estimte
//    estimationPose(model_keypoint,scene_keypoint,model_descriptors_pfh,scene_descriptors_pfh,aligned_model);

	// 可视化
	// visualize_corrs(model_keypoints_shift, scene_keypoints_shift, model, scene, model_scene_corrs);  // 可视化对应关系
    // visualize_Histogram(model_descriptors,scene_descriptors);   // 可视化fpfh直方图

	return 0;
}


