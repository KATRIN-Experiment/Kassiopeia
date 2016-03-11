#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "KEMFileInterface.hh"
#include "KFMDenseBlockSparseMatrixStructure.hh"


#ifdef KEMFIELD_USE_ROOT
#include "TCanvas.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TColor.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TH2D.h"
#endif


using namespace KEMField;

int main(int argc, char** argv)
{

    if(argc != 2){return 1;};

    std::string filename(argv[1]);
    std::cout<<"reading structure file: "<<filename<<std::endl;

    KFMDenseBlockSparseMatrixStructure matrixStructure;

    //read the structure file from disk
    bool result = false;
    KSAObjectInputNode< KFMDenseBlockSparseMatrixStructure >* structure_node;
    structure_node = new KSAObjectInputNode<KFMDenseBlockSparseMatrixStructure>( KSAClassName<KFMDenseBlockSparseMatrixStructure>::name() );
    KEMFileInterface::GetInstance()->ReadKSAFile(structure_node, filename, result);

    if(result)
    {
        matrixStructure = *( structure_node->GetObject() );
        delete structure_node;
    }
    else
    {
        std::cout<<"could not read file"<<std::endl;
        return 1;
    }

    size_t n_blocks = matrixStructure.GetNBlocks();

    unsigned int row_low = 1;
    unsigned int row_high = matrixStructure.GetLargestRowSize();
    unsigned int col_low = 1;
    unsigned int col_high = matrixStructure.GetLargestColumnSize();

    const std::vector<size_t>* rowSizes = matrixStructure.GetRowSizes();
    const std::vector<size_t>* colSizes = matrixStructure.GetColumnSizes();

    #ifdef KEMFIELD_USE_ROOT

    std::cout<<"starting root plotting"<<std::endl;

    //ROOT stuff for plots
    TApplication* App = new TApplication("TestFastMultipoleFielsSolver",&argc,argv);
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

    //plotting objects
    std::vector< TCanvas* > canvas;
    std::vector< TGraph* > graph;
    std::vector< TGraph2D* > graph2d;


    TCanvas* c = new TCanvas("name","name", 50, 50, 950, 850);
    c->SetFillColor(0);
    c->SetRightMargin(0.2);

    TH2D* h = new TH2D("h","block size distribution",(row_high - row_low), row_low, row_high, (col_high - col_low), col_low, col_high);

    for(size_t i=0; i<n_blocks; i++)
    {
        h->Fill(rowSizes->at(i), colSizes->at(i));
    }

    h->Draw("LEGO");

    App->Run();

    #endif


    return 0;
}
