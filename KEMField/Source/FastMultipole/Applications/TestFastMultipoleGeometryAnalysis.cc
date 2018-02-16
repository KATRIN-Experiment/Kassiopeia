#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGBox.hh"
#include "KGRectangle.hh"
#include "KGRotatedObject.hh"
#include "KGMesher.hh"

#include "KGBEM.hh"
#include "KGBEMConverter.hh"

#include "KTypelist.hh"
#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KDataDisplay.hh"

#include "KEMConstants.hh"


#include "KFMElectrostaticSurfaceConverter.hh"
#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainer.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeManager.hh"
#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"
#include "KFMElectrostaticLocalCoefficientCalculatorNumeric.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTreeInformationExtractor.hh"


#include <iostream>
#include <iomanip>



#ifdef KEMFIELD_USE_ROOT
#include "TRandom3.h"
#include "TF1.h"
#include "TComplex.h"
#include "TRandom3.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TColor.h"
#include "TF2.h"
#include "TH1D.h"
#include "TLine.h"
#include "TEllipse.h"
#endif

using namespace KGeoBag;
using namespace KEMField;

int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    int use_box = 0;

    KSurfaceContainer surfaceContainer;

    if(use_box == 1)
    {

        // Construct the shape
        KGBox* box = new KGBox();
        int meshCount = 70;

        box->SetX0(-.5);
        box->SetX1(.5);
        box->SetXMeshCount(meshCount+1);
        box->SetXMeshPower(2);

        box->SetY0(-.5);
        box->SetY1(.5);
        box->SetYMeshCount(meshCount+2);
        box->SetYMeshPower(2);

        box->SetZ0(-.5);
        box->SetZ1(.5);
        box->SetZMeshCount(meshCount+3);
        box->SetZMeshPower(2);

        KGSurface* cube = new KGSurface(box);
        cube->SetName("box");
        cube->MakeExtension<KGMesh>();
        cube->MakeExtension<KGElectrostaticDirichlet>();
        cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        KGMesher* mesher = new KGMesher();
        cube->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        cube->AcceptNode(&geometryConverter);
    }
    else
    {
        int scale = 500;


        // Construct the shape
        double p1[2],p2[2];
        double radius = 1.;
        KGRotatedObject* hemi1 = new KGRotatedObject(scale,20);
        p1[0] = -1.; p1[1] = 0.;
        p2[0] = 0.; p2[1] = 1.;
        hemi1->AddArc(p2,p1,radius,true);

        KGRotatedObject* hemi2 = new KGRotatedObject(scale,20);
        p2[0] = 1.; p2[1] = 0.;
        p1[0] = 0.; p1[1] = 1.;
        hemi2->AddArc(p1,p2,radius,false);

        // Construct shape placement
        KGRotatedSurface* h1 = new KGRotatedSurface(hemi1);
        KGSurface* hemisphere1 = new KGSurface(h1);
        hemisphere1->SetName( "hemisphere1" );
        hemisphere1->MakeExtension<KGMesh>();
        hemisphere1->MakeExtension<KGElectrostaticDirichlet>();
        hemisphere1->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        KGRotatedSurface* h2 = new KGRotatedSurface(hemi2);
        KGSurface* hemisphere2 = new KGSurface(h2);
        hemisphere2->SetName( "hemisphere2" );
        hemisphere2->MakeExtension<KGMesh>();
        hemisphere2->MakeExtension<KGElectrostaticDirichlet>();
        hemisphere2->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        KGMesher* mesher = new KGMesher();
        hemisphere1->AcceptNode(mesher);
        hemisphere2->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        geometryConverter.SetMinimumArea(1.e-12);
        hemisphere1->AcceptNode(&geometryConverter);
        hemisphere2->AcceptNode(&geometryConverter);
    }

    std::cout<<"n elements in surface container = "<<surfaceContainer.size()<<std::endl;


    //now we have a surface container with a bunch of electrode discretizations
    //we just want to convert these into point clouds, and then bounding balls
    //extract the information we want
    KFMElectrostaticElementContainer<3,1>* elementContainer = new KFMElectrostaticElementContainer<3,1>();
    KFMElectrostaticSurfaceConverter* converter = new KFMElectrostaticSurfaceConverter();
    converter->SetSurfaceContainer(&surfaceContainer);
    converter->SetElectrostaticElementContainer(elementContainer);
    converter->Extract();

    //tree construction parameters
    KFMElectrostaticParameters params;
    params.divisions = 3;
    params.degree = 0;
    params.zeromask = 1;
    params.maximum_tree_depth = 4;
    params.region_expansion_factor = 2.1;
    params.use_region_estimation = true;


    //build the tree
    KFMElectrostaticTree* e_tree = new KFMElectrostaticTree();
    //create the tree manager
    KFMElectrostaticTreeManager<KFMElectrostaticTreeManager_SingleThread>* treeManager = new KFMElectrostaticTreeManager<KFMElectrostaticTreeManager_SingleThread>();
    treeManager->SetElectrostaticElementContainer(elementContainer);
    treeManager->SetTree(e_tree);
    treeManager->SetParameters(params);
    treeManager->Initialize();

    //build the tree and node objects
    treeManager->ConstructRootNode();
    std::cout<<"done constructing root node"<<std::endl;
    treeManager->PerformSpatialSubdivision();
    std::cout<<"done subdivision"<<std::endl;
    treeManager->AssociateElementsAndNodes();
    std::cout<<"done element node association"<<std::endl;
    treeManager->RemoveMultipoleMoments();
    std::cout<<"done removing any pre-existing multipole moments"<<std::endl;
    treeManager->ComputeMultipoleMoments();
    std::cout<<"done computing multipole moments"<<std::endl;
    treeManager->PerformAdjacencySubdivision();




    #ifdef KEMFIELD_USE_ROOT
    //ROOT stuff for plots
    TApplication* App = new TApplication("ERR",&argc,argv);
    TStyle* myStyle = new TStyle("Plain", "Plain");
    myStyle->SetCanvasBorderMode(0);
    myStyle->SetPadBorderMode(0);
    myStyle->SetPadColor(0);
    myStyle->SetCanvasColor(0);
    myStyle->SetTitleColor(1);
    myStyle->SetPalette(1,0);   // nice color scale for z-axis
    myStyle->SetCanvasBorderMode(0); // gets rid of the stupid raised edge around the canvas
    myStyle->SetTitleFillColor(0); //turns the default dove-grey background to white
    myStyle->SetCanvasColor(0);
    myStyle->SetPadColor(0);
    myStyle->SetTitleFillColor(0);
    myStyle->SetStatColor(0); //this one may not work
    const int NRGBs = 5;
    const int NCont = 48;
    double stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    double red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
    double green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
    double blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    myStyle->SetNumberContours(NCont);
    myStyle->cd();


    TCanvas* canvas = new TCanvas("potential","potential", 50, 50, 950, 850);
    canvas->Divide(3,3);
    canvas->SetFillColor(0);
    canvas->SetRightMargin(0.2);

    KFMElectrostaticElementContainer<3,1>* container2 = new KFMElectrostaticElementContainer<3,1>();
    KFMElectrostaticTreeInformationExtractor extractor;

    e_tree->ApplyRecursiveAction(&extractor);

    std::vector<KFMIdentitySet>* id_set_lists = extractor.GetLevelIDSets();

    extractor.PrintStatistics();

    //loop over all bounding balls in geometry and histogram the sizes

    canvas->cd(1);
    TH1* h  = new TH1D("h1", "h1 title", 1000, 1e-8, 0.02);

    unsigned int n = elementContainer->GetNElements();
    double radius;
    for(unsigned int i=0; i<n; i++)
    {
        radius = elementContainer->GetBoundingBall(i)->GetRadius();
        h->Fill(radius);
    }

    h->Draw();

    TH1* histo[9];

    std::vector<unsigned int> id_list;
    for(unsigned int i=0; i<id_set_lists->size(); i++)
    {
        canvas->cd(i+1);
        std::stringstream ss;
        ss<<"histo ";
        ss<<i;
        histo[i] = new TH1D(ss.str().c_str(), ss.str().c_str(), 1000, 1e-8, 0.02);
        histo[i]->SetFillColor(i+1);

        id_list.clear();
        id_set_lists->at(i).GetIDs(&id_list);

        for(unsigned int j=0; j<id_list.size(); j++)
        {
            radius= elementContainer->GetBoundingBall(id_list[j])->GetRadius();
            histo[i]->Fill(radius);

            //not at bottom level so add to secondary container
            //if(i != params.maximum_tree_depth)
            {
                container2->AddElectrostaticElement(elementContainer->GetElectrostaticElement(id_list[j]));
            }
        }

        histo[i]->Draw("SAME");
    }


    KFMElectrostaticNode* root = e_tree->GetRootNode();

    KFMPoint<3> center_point = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(root)->GetCenter();
    double length = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(root)->GetLength();

    //tree construction parameters
    KFMElectrostaticParameters params2;
    params2.divisions = 2;
    params2.degree = 0;
    params2.zeromask = 1;
    params2.maximum_tree_depth = 8;
    params2.region_expansion_factor = 2.1;
    params2.use_region_estimation = false;
    params2.world_center_x = center_point[0];
    params2.world_center_y = center_point[1];
    params2.world_center_z = center_point[2];
    params2.world_length = length;


    //build the tree
    KFMElectrostaticTree* e_tree2 = new KFMElectrostaticTree();
    //create the tree manager
    KFMElectrostaticTreeManager<KFMElectrostaticTreeManager_SingleThread>* treeManager2 = new KFMElectrostaticTreeManager<KFMElectrostaticTreeManager_SingleThread>();
    treeManager2->SetElectrostaticElementContainer(container2);
    treeManager2->SetTree(e_tree2);
    treeManager2->SetParameters(params2);
    treeManager2->Initialize();

    //build the tree and node objects
    treeManager2->ConstructRootNode();
    std::cout<<"done constructing root node"<<std::endl;
    treeManager2->PerformSpatialSubdivision();
    std::cout<<"done subdivision"<<std::endl;
    treeManager2->AssociateElementsAndNodes();
    std::cout<<"done element node association"<<std::endl;
    treeManager2->RemoveMultipoleMoments();
    std::cout<<"done removing any pre-existing multipole moments"<<std::endl;
    treeManager2->ComputeMultipoleMoments();
    std::cout<<"done computing multipole moments"<<std::endl;
    treeManager2->PerformAdjacencySubdivision();


    TCanvas* canvas2 = new TCanvas("2","2", 50, 50, 950, 850);
    canvas2->Divide(3,3);
    canvas2->SetFillColor(0);
    canvas2->SetRightMargin(0.2);


    KFMElectrostaticTreeInformationExtractor extractor2;

    e_tree2->ApplyRecursiveAction(&extractor2);

    std::vector<KFMIdentitySet>* id_set_lists2 = extractor2.GetLevelIDSets();

    //loop over all bounding balls in geometry and histogram the sizes

    extractor2.PrintStatistics();

    canvas2->cd(1);
    TH1* h2  = new TH1D("h2", "h2 title", 1000, 1e-8, 0.02);

    unsigned int n2 = container2->GetNElements();
    for(unsigned int i=0; i<n2; i++)
    {
        radius = container2->GetBoundingBall(i)->GetRadius();
        h2->Fill(radius);
    }

    h2->Draw();

    TH1* histo2[10];

    std::vector<unsigned int> id_list2;
    for(unsigned int i=0; i<id_set_lists2->size(); i++)
    {
        canvas2->cd(i+1);
        std::stringstream ss;
        ss<<"histo2 ";
        ss<<i;
        histo2[i] = new TH1D(ss.str().c_str(), ss.str().c_str(), 1000, 1e-8, 0.02);
        histo2[i]->SetFillColor(i+1);

        id_set_lists2->at(i).GetIDs(&id_list2);

        for(unsigned int j=0; j<id_list2.size(); j++)
        {
            radius= container2->GetBoundingBall(id_list2[j])->GetRadius();
            histo2[i]->Fill(radius);
        }

        histo2[i]->Draw("SAME");
    }





















    canvas->Update();
    App->Run();

    #endif



























//    delete e_tree;
//    delete treeManager;
//    delete calc;

    return 0;
}
