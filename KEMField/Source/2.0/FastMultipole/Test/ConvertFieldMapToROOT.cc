#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"
#include "KEMFileInterface.hh"


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
#include "TLine.h"
#include "TEllipse.h"
#include "TGraph.h"
#endif

using namespace KEMField;

int main(int argc, char** argv)
{

    if(argc != 2)
    {
        std::cout<<"please give path to file"<<std::endl;
        return 1;
    }

    std::string filename(argv[1]);

    KSAObjectInputNode< KFMNamedScalarDataCollection >* data_node = new KSAObjectInputNode< KFMNamedScalarDataCollection >("data_collection");

    bool result;
    KEMFileInterface::GetInstance()->ReadKSAFile(data_node, filename, result);

    if(!result)
    {
        std::cout<<"failed to read file"<<std::endl;
        return 1;
    };

    KFMNamedScalarDataCollection* data  = data_node->GetObject();

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

    KFMNamedScalarData* x_coord = data->GetDataWithName(std::string("x_coordinate"));
    KFMNamedScalarData* y_coord = data->GetDataWithName(std::string("y_coordinate"));
    KFMNamedScalarData* z_coord = data->GetDataWithName(std::string("z_coordinate"));

    #ifdef KEMFIELD_USE_ROOT
    //ROOT stuff for plots
    std::vector<TGraph*> graph;
    std::vector<TCanvas*> canvas;

    TApplication* App = new TApplication("field_map",&argc,argv);
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

    //collect the rest of the (non-point) scalar data
    unsigned int n_data_sets = data->GetNDataSets();
    for(unsigned int i=0; i<n_data_sets; i++)
    {
        std::string name = data->GetDataSetWithIndex(i)->GetName();

        std::cout<<"scanning: "<<name<<std::endl;
        if( (name != (std::string("x_coordinate") ) ) && (name != (std::string("y_coordinate") ) ) && (name != (std::string("z_coordinate") ) ) )
        {
            if( (name != (std::string("fmm_time_per_potential_call") ) ) &&
                (name != (std::string("fmm_time_per_field_call") ) ) &&
                (name != (std::string("direct_time_per_potential_call") ) ) &&
                (name != (std::string("direct_time_per_field_call") ) )   )
            {

                unsigned int size = data->GetDataSetWithIndex(i)->GetSize();
                if(size == z_coord->GetSize())
                {
                    std::cout<<"making graph for "<<name<<std::endl;

                    TCanvas* c = new TCanvas(name.c_str(),name.c_str(), 50, 50, 950, 850);
                    c->SetFillColor(0);
                    c->SetRightMargin(0.2);


                    TGraph* g = new TGraph();
                    g->SetTitle(name.c_str());
                    graph.push_back(g);

                    for(unsigned int j=0; j<size; j++)
                    {
                        g->SetPoint(j, z_coord->GetValue(j), data->GetDataSetWithIndex(i)->GetValue(j) );
                    }

                    g->Draw("ALP");
                    c->Update();

                    canvas.push_back(c);
                }
            }
        }
    }

    std::cout<<"starting root app"<<std::endl;

    App->Run();

    #else
        std::cout<<"Please re-compile with ROOT support enabled to use this program."<<std::endl;
    #endif

    return 0;
}
