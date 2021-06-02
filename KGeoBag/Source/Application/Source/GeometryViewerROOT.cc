#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGROOTGeometryPainter.hh"
#include "KROOTWindow.h"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout
            << "usage: ./GeometryPainterROOT <config_file_name.xml> <geometry_path> [...] <plane normal vector> <plane point> <swap axis>"
            << "     : ./GeometryPainterROOT <config_file_name.xml> <geometry_path> [...] --plane={XZ,YZ,XY} [--point='<plane point>'] [--labels=true]"
            << endl;
        return -1;
    }

    coremsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv, true);

    deque<string> tParameters = tXML.GetArguments().ParameterList();
    tParameters.pop_front();  // strip off config file name

    KThreeVector tPlaneNormal(0,1,0);  // default to yz-plane
    KThreeVector tPlanePoint(0,0,0);
    bool tSwapAxes = false;
    bool tShowLabels = false;

    auto tOptions = tXML.GetArguments().OptionTable();
    if (tOptions.find("plane") != tOptions.end()) {
        if (tOptions["plane"] == "ZX" || tOptions["plane"] == "zx") {
            tPlaneNormal = KThreeVector(1, 0, 0);
        }
        else if (tOptions["plane"] == "XZ" || tOptions["plane"] == "xz") {
            tPlaneNormal = KThreeVector(1, 0, 0);
            tSwapAxes = true;
        }
        else if (tOptions["plane"] == "YZ" || tOptions["plane"] == "yz") {
            tPlaneNormal = KThreeVector(0 , 0, 0);
        }
        else if (tOptions["plane"] == "ZY" || tOptions["plane"] == "zy") {
            tPlaneNormal = KThreeVector(0 , 0, 0);
            tSwapAxes = true;
        }
        else if (tOptions["plane"] == "YX" || tOptions["plane"] == "yx") {
            tPlaneNormal = KThreeVector(0 , 0, 1);
        }
        else if (tOptions["plane"] == "XY" || tOptions["plane"] == "xy") {
            tPlaneNormal = KThreeVector(0 , 0, 1);
            tSwapAxes = true;
        }
        else
            coremsg(eError) << "plane definition <" << tOptions["plane"] << "> is not supported" << eom;
    }
    if (tOptions.find("point") != tOptions.end()) {
        istringstream Converter(tOptions["point"]);
        Converter >> tPlanePoint;
    }
    if (tOptions.find("labels") != tOptions.end()) {
        tShowLabels = true;
    }

    deque<string> tPathList;  // note 7 more following arguments after paths
    if (tParameters.size() > 7) {
        while (tParameters.size() > 7) {
            tPathList.push_back(tParameters.front());
            tParameters.pop_front();
        }
    }
    else {
        tPathList = tParameters;
    }

    coremsg(eNormal) << "...initialization finished" << eom;

    KROOTWindow tWindow;
    tWindow.SetName("KGeoBag ROOT Geometry Viewer");

    KGROOTGeometryPainter tPainter;
    tPainter.SetName("ROOT GeometryPainter");
    tPainter.SetDisplayMode(true);
    tPainter.SetWriteMode(true);
    tPainter.SetShowLabels(tShowLabels);

    if (tParameters.size() >= 3) {
        string tNormalX(tParameters[0]);
        string tNormalY(tParameters[1]);
        string tNormalZ(tParameters[2]);
        string tSpaceString(" ");
        string tCombine = tNormalX + tSpaceString + tNormalY + tSpaceString + tNormalZ;
        istringstream Converter(tCombine);
        Converter >> tPlaneNormal;
    }

    if (tParameters.size() >= 6) {
        string tX(tParameters[3]);
        string tY(tParameters[4]);
        string tZ(tParameters[5]);
        string tSpaceString(" ");
        string tCombine = tX + tSpaceString + tY + tSpaceString + tZ;
        istringstream Converter(tCombine);
        Converter >> tPlanePoint;
    }

    if (tParameters.size() >= 7) {
        if (tParameters[6] == string("true") || tParameters[6] == string("1")) {
            tSwapAxes = true;
        }
        else if (tParameters[6] == string("false") || tParameters[6] == string("0")) {
            tSwapAxes = false;
        }
    }

    tPainter.SetPlaneNormal(tPlaneNormal);
    tPainter.SetPlanePoint(tPlanePoint);
    tPainter.SetSwapAxis(tSwapAxes);

    coremsg(eNormal) << "painting with plane normal " << tPlaneNormal << ", plane point " << tPlanePoint << eom;

    for (auto& tPath : tPathList) {
        for (auto& tSurface : KGInterface::GetInstance()->RetrieveSurfaces(tPath)) {
            tPainter.AddSurface(tSurface);
        }
        for (auto& tSpace : KGInterface::GetInstance()->RetrieveSpaces(tPath)) {
            tPainter.AddSpace(tSpace);
        }
    }

    tWindow.AddPainter(&tPainter);
    tWindow.Render();
    tWindow.Display();
    tWindow.Write();
    tWindow.RemovePainter(&tPainter);

    return 0;
}
