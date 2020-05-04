#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticTreeData.hh"
#include "KFMVTKElectrostaticTreeViewer.hh"
#include "KSADataStreamer.hh"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace KEMField;

int main(int argc, char* argv[])
{
    std::string usage = "\n"
                        "Usage: VisualizeElectrostaticMultipoleTree <Options> <TreeFile> <ParaViewFile>\n"
                        "\n"
                        "This program takes a KEMField geometry file and produces two ParaView files of the\n"
                        "geometry ('<ParaViewFile>.vtp') and the multipole tree ('<ParaViewFile>.vtu').\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -n, --name               (required; name of tree, e.g from InspectEMFile)\n"
                        "\t -d, --display            (bool; display geometry on screen)\n";

    if (argc <= 2)  // --name is a required argument
    {
        std::cout << usage;
        return 0;
    }

    static struct option longOptions[] = {
        {"help", no_argument, 0, 'h'},
        {"name", required_argument, 0, 'n'},
        {"display", no_argument, 0, 'd'},
    };

    static const char* optString = "hn:";

    // create label set for multipole tree container object
    string name = KFMElectrostaticTreeData::Name();
    bool display = false;

    while (1) {
        char optId = getopt_long(argc, argv, optString, longOptions, NULL);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('n'):
                name = std::string(optarg);
                break;
            case ('d'):
                display = true;
                break;
            default:  // unrecognized option
                std::cout << usage << std::endl;
                return 1;
        }
    }

    std::string inFileName = argv[optind];
    std::string suffix = inFileName.substr(inFileName.find_last_of("."), std::string::npos);

    std::string outFileName = inFileName.substr(0, inFileName.find_last_of("."));
    if (optind != argc - 1)
        outFileName = argv[optind + 1];

    KFMElectrostaticTreeData* tree_data = new KFMElectrostaticTreeData();
    KFMElectrostaticTree* tree = new KFMElectrostaticTree();

    KEMFileInterface::GetInstance()->Read(inFileName, *tree_data, name);

    std::cout << "Fast multipole tree found." << std::endl;

    //construct tree from data
    KFMElectrostaticTreeConstructor<> constructor;
    constructor.ConstructTree(*tree_data, *tree);

    KFMVTKElectrostaticTreeViewer viewer(*tree);

    viewer.GenerateGeometryFile(outFileName + std::string(".vtp"));
    viewer.GenerateGridFile(outFileName + std::string(".vtu"));
    if (display)
        viewer.ViewGeometry();

    return 0;
}
