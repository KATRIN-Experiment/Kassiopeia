#include "KBinaryDataStreamer.hh"
#include "KDataDisplay.hh"
#include "KEMConstants.hh"
#include "KEMFileInterface.hh"
#include "KEMTransformation.hh"
#include "KEMVTKViewer.hh"
#include "KSADataStreamer.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KTypelist.hh"

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
                        "Usage: VisualizeEMGeometry <GeometryFile> <ParaViewFile>\n"
                        "\n"
                        "This program takes a KEMField geometry file and produces a ParaView file of the\n"
                        "geometry ('<ParaViewFile>.vtp').\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -n, --name               (required; name of surface container, e.g from InspectEMFile)\n"
                        "\t -d, --display            (bool; display geometry on screen)\n";

    if (argc <= 2)  // --name is a required argument
    {
        std::cout << usage;
        return 0;
    }

    static struct option longOptions[] = {
        {"help", no_argument, nullptr, 'h'},
        {"name", required_argument, nullptr, 'n'},
        {"display", no_argument, nullptr, 'd'},
    };

    static const char* optString = "hn:d";

    std::string name = KSurfaceContainer::Name();
    bool display = false;

    while (true) {
        int optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('n'):
                name = optarg;
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
    std::string suffix = inFileName.substr(inFileName.find_last_of('.'), std::string::npos);

    std::string outFileName = inFileName.substr(0, inFileName.find_last_of('.'));
    if (optind != argc - 1)
        outFileName = argv[optind + 1];

    KSurfaceContainer surfaceContainer;

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(inFileName.c_str(), &fileInfo);
    if (fileStat == 0)
        exists = true;
    else
        exists = false;

    if (!exists) {
        std::cout << "Error: file \"" << inFileName << "\" cannot be read." << std::endl;
        return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (suffix != binaryDataStreamer.GetFileSuffix()) {
        std::cout << "Error: unkown file extension \"" << suffix << "\"" << std::endl;
        return 1;
    }

    KEMFileInterface::GetInstance()->Read(inFileName, surfaceContainer, name);

    KEMVTKViewer viewer(surfaceContainer);

    viewer.GenerateGeometryFile(outFileName + std::string(".vtp"));
    if (display)
        viewer.ViewGeometry();

    return 0;
}
