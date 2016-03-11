#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>

#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeData.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"

#include "KFMVTKElectrostaticTreeViewer.hh"

using namespace KEMField;

int main(int argc, char* argv[])
{
    std::string usage =
    "\n"
    "Usage: VisualizeElectrostaticMultipoleTree <Options> <TreeFile> <ParaViewFile>\n"
    "\n"
    "This program takes a KEMField geometry file and produces a ParaView file of the\n"
    "geometry.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -n, --name               (name of tree )\n";

    if(argc == 1)
    {
        std::cout<<usage;
        return 0;
    }

    static struct option longOptions[] =
    {
        {"help", no_argument, 0, 'h'},
        {"name", required_argument, 0, 'n'}
    };

    static const char *optString = "hn:";

    // create label set for multipole tree container object
    string name = KFMElectrostaticTreeData::Name();

    while(1)
    {
        char optId = getopt_long(argc, argv,optString, longOptions, NULL);
        if(optId == -1) break;
        switch(optId)
        {
            case('h'): // help
            std::cout<<usage<<std::endl;
            return 0;
            case('n'):
            name = std::string(optarg);
            break;
            default: // unrecognized option
            std::cout<<usage<<std::endl;
            return 1;
        }
    }

    std::string inFileName = argv[optind];
    std::string suffix = inFileName.substr(inFileName.find_last_of("."),std::string::npos);

    std::string outFileName = inFileName.substr(0,inFileName.find_last_of(".")) + std::string(".vtp");
    if (optind != argc-1)
    outFileName = argv[optind+1];

    KFMElectrostaticTreeData* tree_data = new KFMElectrostaticTreeData();
    KFMElectrostaticTree* tree = new KFMElectrostaticTree();

    KEMFileInterface::GetInstance()->Read(inFileName,*tree_data,name);

    std::cout << "Fast multipole tree found." << std::endl;

    //construct tree from data
    KFMElectrostaticTreeConstructor<> constructor;
    constructor.ConstructTree(*tree_data, *tree);

    KFMVTKElectrostaticTreeViewer viewer(*tree);

    viewer.GenerateGeometryFile(outFileName);

    return 0;
}
