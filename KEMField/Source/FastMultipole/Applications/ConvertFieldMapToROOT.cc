#include "KEMFileInterface.hh"
#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#ifdef KEMFIELD_USE_ROOT
#include "TApplication.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TComplex.h"
#include "TEllipse.h"
#include "TF1.h"
#include "TF2.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TLine.h"
#include "TRandom3.h"
#include "TStyle.h"
#endif

using namespace KEMField;

int main(int argc, char** argv)
{

    std::string usage = "\n"
                        "Usage: ConvertFieldMapToROOT <fieldmap file> <options>\n"
                        "\n"
                        "This program plots previously evaluated field data as a function of coordinates. \n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -f  --file               (full path to fieldmap file)\n"
                        "\t -x, --x_coord            (plot as a function of x coordinate)\n"
                        "\t -y, --y_coord            (plot as a function of y coordinate)\n"
                        "\t -z, --z_coord            (plot as a function of z coordinate)\n";

    std::string file;
    unsigned int mode = 0;
    bool use_x = false;
    bool use_y = false;
    bool use_z = false;

    static struct option longOptions[] = {
        {"help", no_argument, nullptr, 'h'},
        {"file", required_argument, nullptr, 'f'},
        {"x_coord", no_argument, nullptr, 'x'},
        {"y_coord", no_argument, nullptr, 'y'},
        {"z_coord", no_argument, nullptr, 'z'},
    };

    static const char* optString = "hf:xyz";

    while (true) {
        char optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('f'):
                file = std::string(optarg);
                break;
            case ('x'):
                use_x = true;
                break;
            case ('y'):
                use_y = true;
                break;
            case ('z'):
                use_z = true;
                break;
            default:
                std::cout << usage << std::endl;
                return 1;
        }
    }


    //determine usage case
    if (use_x && use_y && use_z) {
        mode = 0;
    };  //histogram all data
    if (!use_x && !use_y && !use_z) {
        mode = 0;
    };  //histogram all data
    if (use_x && !use_y && !use_z) {
        mode = 1;
    };  //1d plot as function of x
    if (!use_x && use_y && !use_z) {
        mode = 2;
    };  //1d plot as function of y
    if (!use_x && !use_y && use_z) {
        mode = 3;
    };  //1d plot as function of z
    if (use_x && use_y && !use_z) {
        mode = 4;
    };  //2d plot as function of (x,y)
    if (!use_x && use_y && use_z) {
        mode = 5;
    };  //2d plot as function of (y,z)
    if (use_x && !use_y && use_z) {
        mode = 6;
    };  //2d plot as function of (x,z)

    auto* data_node = new KSAObjectInputNode<KFMNamedScalarDataCollection>("data_collection");

    bool result;
    KEMFileInterface::GetInstance()->ReadKSAFile(data_node, file, result);

    if (!result) {
        std::cout << "failed to read file" << std::endl;
        return 1;
    };

    KFMNamedScalarDataCollection* data = data_node->GetObject();

    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    KFMNamedScalarData* x_coord = nullptr;
    KFMNamedScalarData* y_coord = nullptr;
    KFMNamedScalarData* z_coord = nullptr;

    if (use_x) {
        x_coord = data->GetDataWithName(std::string("x_coordinate"));
    }

    if (use_y) {
        y_coord = data->GetDataWithName(std::string("y_coordinate"));
    }

    if (use_z) {
        z_coord = data->GetDataWithName(std::string("z_coordinate"));
    }

#ifdef KEMFIELD_USE_ROOT

    //ROOT stuff for plots
    std::vector<TGraph*> graph;
    std::vector<TGraph2D*> graph2d;
    std::vector<TH1D*> histo;
    std::vector<TCanvas*> canvas;

    TApplication* App = new TApplication("field_map", &argc, argv);
    TStyle* myStyle = new TStyle("Plain", "Plain");
    myStyle->SetCanvasBorderMode(0);
    myStyle->SetPadBorderMode(0);
    myStyle->SetPadColor(0);
    myStyle->SetCanvasColor(0);
    myStyle->SetTitleColor(1);
    myStyle->SetPalette(1, 0);        // nice color scale for z-axis
    myStyle->SetCanvasBorderMode(0);  // gets rid of the stupid raised edge around the canvas
    myStyle->SetTitleFillColor(0);    //turns the default dove-grey background to white
    myStyle->SetCanvasColor(0);
    myStyle->SetPadColor(0);
    myStyle->SetTitleFillColor(0);
    myStyle->SetStatColor(0);  //this one may not work
    const int NRGBs = 5;
    const int NCont = 48;
    double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
    double red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
    double green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
    double blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    myStyle->SetNumberContours(NCont);
    myStyle->cd();

    //collect the rest of the (non-point) scalar data
    unsigned int n_data_sets = data->GetNDataSets();
    for (unsigned int i = 0; i < n_data_sets; i++) {
        std::string name = data->GetDataSetWithIndex(i)->GetName();

        std::cout << "scanning: " << name << std::endl;
        if ((name != (std::string("x_coordinate"))) && (name != (std::string("y_coordinate"))) &&
            (name != (std::string("z_coordinate")))) {
            if ((name != (std::string("fmm_time_per_potential_call"))) &&
                (name != (std::string("fmm_time_per_field_call"))) &&
                (name != (std::string("direct_time_per_potential_call"))) &&
                (name != (std::string("direct_time_per_field_call")))) {
                unsigned int size = data->GetDataSetWithIndex(i)->GetSize();
                std::cout << "making graph for " << name << " with size: " << size << std::endl;

                TCanvas* c = new TCanvas(name.c_str(), name.c_str(), 50, 50, 950, 850);
                c->SetFillColor(0);
                c->SetRightMargin(0.2);

                if (mode == 0) {
                    //TODO add histograming option
                    // std::string title = name + " histogram";
                    // TH1D* h = new TH1D();
                    // h->SetTitle(name.c_str());
                    // histo.push_back(h);
                    // c->Update();
                    // canvas.push_back(c);
                }

                if (mode == 1) {
                    std::string title = name + " vs. x";
                    TGraph* g = new TGraph();
                    g->SetTitle(name.c_str());
                    graph.push_back(g);
                    for (unsigned int j = 0; j < size; j++) {
                        g->SetPoint(j, x_coord->GetValue(j), data->GetDataSetWithIndex(i)->GetValue(j));
                    }
                    g->Draw("ALP");
                    c->Update();
                    canvas.push_back(c);
                }
                if (mode == 2) {
                    std::string title = name + " vs. y";
                    TGraph* g = new TGraph();
                    g->SetTitle(name.c_str());
                    graph.push_back(g);
                    for (unsigned int j = 0; j < size; j++) {
                        g->SetPoint(j, y_coord->GetValue(j), data->GetDataSetWithIndex(i)->GetValue(j));
                    }
                    g->Draw("ALP");
                    c->Update();
                    canvas.push_back(c);
                }
                if (mode == 3) {
                    std::string title = name + " vs. z";
                    TGraph* g = new TGraph();
                    g->SetTitle(name.c_str());
                    graph.push_back(g);
                    for (unsigned int j = 0; j < size; j++) {
                        g->SetPoint(j, z_coord->GetValue(j), data->GetDataSetWithIndex(i)->GetValue(j));
                    }
                    g->Draw("ALP");
                    c->Update();
                    canvas.push_back(c);
                }

                if (mode == 4) {
                    TGraph2D* g = new TGraph2D(size);
                    std::string title = name + " in x-y plane";
                    g->SetTitle(title.c_str());
                    graph2d.push_back(g);
                    for (unsigned int j = 0; j < size; j++) {
                        g->SetPoint(j,
                                    x_coord->GetValue(j),
                                    y_coord->GetValue(j),
                                    data->GetDataSetWithIndex(i)->GetValue(j));
                    }
                    g->SetMarkerStyle(24);
                    g->Draw("PCOLZ");
                    c->Update();
                    canvas.push_back(c);
                }

                if (mode == 5) {
                    TGraph2D* g = new TGraph2D(size);
                    std::string title = name + " in y-z plane";
                    g->SetTitle(title.c_str());
                    graph2d.push_back(g);
                    for (unsigned int j = 0; j < size; j++) {
                        g->SetPoint(j,
                                    y_coord->GetValue(j),
                                    z_coord->GetValue(j),
                                    data->GetDataSetWithIndex(i)->GetValue(j));
                    }
                    g->SetMarkerStyle(24);
                    g->Draw("PCOLZ");
                    c->Update();
                    canvas.push_back(c);
                }
                if (mode == 6) {
                    TGraph2D* g = new TGraph2D(size);
                    std::string title = name + " in x-z plane";
                    g->SetTitle(title.c_str());
                    graph2d.push_back(g);
                    for (unsigned int j = 0; j < size; j++) {
                        g->SetPoint(j,
                                    x_coord->GetValue(j),
                                    z_coord->GetValue(j),
                                    data->GetDataSetWithIndex(i)->GetValue(j));
                    }
                    g->SetMarkerStyle(24);
                    g->Draw("PCOLZ");
                    c->Update();
                    canvas.push_back(c);
                }
            }
        }
    }

    std::cout << "starting root app" << std::endl;

    App->Run();

#else
    std::cout << "Please re-compile with ROOT support enabled to use this program." << std::endl;
#endif

    return 0;
}
