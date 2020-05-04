#include "KCommandLineTokenizer.hh"
#include "KConditionProcessor.hh"
#include "KElementProcessor.hh"
#include "KGBox.hh"
#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGMeshElementCollector.hh"
#include "KGMesher.hh"
#include "KGNavigableMeshElementContainer.hh"
#include "KGNavigableMeshTree.hh"
#include "KGNavigableMeshTreeBuilder.hh"
#include "KGRectangle.hh"
#include "KGRotatedObject.hh"
#include "KGVTKMeshIntersectionTester.hh"
#include "KGVTKMeshPainter.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KMessage.h"
#include "KPrintProcessor.hh"
#include "KTagProcessor.hh"
#include "KTextFile.h"
#include "KVTKWindow.h"
#include "KVariableProcessor.hh"
#include "KXMLTokenizer.hh"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace KGeoBag;


int main(int argc, char** argv)
{
    std::string usage =
        "\n"
        "Usage: TestMeshNavigation <options>\n"
        "\n"
        "This program computes the solution of a simple dirichlet problem and compares the fast multipole field to the direct sovler. \n"
        "\tAvailable options:\n"
        "\t -h, --help               (shows this message and exits)\n"
        "\t -s, --scale              (scale of geometry discretization)\n"
        "\t -g, --geometry           (0: Single sphere, \n"
        "\t                           1: Single cube, \n"
        "\t                           2: Cube and sphere \n";

    unsigned int scale = 20;
    unsigned int geometry = 0;

    static struct option longOptions[] = {{"help", no_argument, 0, 'h'},
                                          {"scale", required_argument, 0, 's'},
                                          {"geometry", required_argument, 0, 'g'}};

    static const char* optString = "hs:g:";

    while (1) {
        char optId = getopt_long(argc, argv, optString, longOptions, NULL);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('s'):
                scale = atoi(optarg);
                break;
            case ('g'):
                geometry = atoi(optarg);
                break;
            default:
                std::cout << usage << std::endl;
                return 1;
        }
    }

    //we need a container to store all of the mesh elements


    vector<KGSurface*> tSurfaces;
    vector<KGSurface*>::iterator tSurfaceIt;


    if (geometry == 0 || geometry == 2) {
        // Construct the shape
        double p1[2], p2[2];
        double radius = 1.;
        KGRotatedObject* hemi1 = new KGRotatedObject(scale, 20);
        p1[0] = -1.;
        p1[1] = 0.;
        p2[0] = 0.;
        p2[1] = 1.;
        hemi1->AddArc(p2, p1, radius, true);

        KGRotatedObject* hemi2 = new KGRotatedObject(scale, 20);
        p2[0] = 1.;
        p2[1] = 0.;
        p1[0] = 0.;
        p1[1] = 1.;
        hemi2->AddArc(p1, p2, radius, false);

        // Construct shape placement
        KGRotatedSurface* h1 = new KGRotatedSurface(hemi1);
        KGSurface* hemisphere1 = new KGSurface(h1);
        hemisphere1->SetName("hemisphere1");
        hemisphere1->MakeExtension<KGMesh>();

        KGRotatedSurface* h2 = new KGRotatedSurface(hemi2);
        KGSurface* hemisphere2 = new KGSurface(h2);
        hemisphere2->SetName("hemisphere2");
        hemisphere2->MakeExtension<KGMesh>();

        tSurfaces.push_back(hemisphere1);
        tSurfaces.push_back(hemisphere2);
    }

    if (geometry == 1 || geometry == 2) {

        // Construct the shape
        KGBox* box = new KGBox();
        int meshCount = scale;

        box->SetX0(-.5);
        box->SetX1(.5);
        box->SetXMeshCount(meshCount);
        box->SetXMeshPower(3);

        box->SetY0(-.5);
        box->SetY1(.5);
        box->SetYMeshCount(meshCount);
        box->SetYMeshPower(3);

        box->SetZ0(-.5);
        box->SetZ1(.5);
        box->SetZMeshCount(meshCount);
        box->SetZMeshPower(3);

        KGSurface* cube = new KGSurface(box);
        cube->SetName("box");
        cube->MakeExtension<KGMesh>();

        tSurfaces.push_back(cube);
    }


    KGNavigableMeshElementContainer tContainer;

    KGMeshElementCollector tCollector;
    tCollector.SetMeshElementContainer(&tContainer);


    KVTKWindow tWindow;
    tWindow.SetName("KGeoBag Mesh Viewer");
    tWindow.SetFrameColorRed(0.);
    tWindow.SetFrameColorGreen(0.);
    tWindow.SetFrameColorBlue(0.);
    tWindow.SetDisplayMode(true);
    tWindow.SetWriteMode(true);

    KGMesher tMesher;

    KGVTKMeshPainter tPainter;
    tPainter.SetName("MeshPainter");
    tPainter.SetDisplayMode(true);
    tPainter.SetWriteMode(true);
    tPainter.SetColorMode(KGVTKMeshPainter::sModulo);

    for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
        (*tSurfaceIt)->MakeExtension<KGMesh>();
        (*tSurfaceIt)->AcceptNode(&tMesher);
        (*tSurfaceIt)->AcceptNode(&tPainter);
        (*tSurfaceIt)->AcceptNode(&tCollector);
    }

    KGVTKMeshIntersectionTester tTester;
    tTester.SetName("MeshIntersectionTester");
    tTester.SetDisplayMode(true);
    tTester.SetWriteMode(true);
    tTester.SetSampleCount(1000);
    tTester.SetSampleColor(KGRGBColor(127, 127, 127));
    tTester.SetUnintersectedLineColor(KGRGBColor(127, 127, 127));
    tTester.SetUnintersectedLineColor(KGRGBColor(0, 0, 255));
    tTester.SetVertexSize(0.001);
    tTester.SetLineSize(0.001);


    for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
        tTester.AddSurface(*tSurfaceIt);
    }

    tWindow.AddPainter(&tPainter);
    tWindow.AddPainter(&tTester);

    tWindow.Render();
    tWindow.Write();
    tWindow.Display();
    tWindow.RemovePainter(&tPainter);
    tWindow.RemovePainter(&tTester);
}
