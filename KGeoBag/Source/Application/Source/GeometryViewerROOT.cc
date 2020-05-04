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
    if (argc < 10) {
        cout
            << "usage: ./GeometryPainterROOT <config_file_name.xml> <geometry_path> [...] <plane normal vector> <plane point> <swap axis>"
            << endl;
        return -1;
    }

    coremsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv, true);

    deque<string> tParameters = tXML.GetArguments().ParameterList();
    tParameters.pop_front();  // strip off config file name

    vector<string> tPathList;  // note 7 more following arguments after paths
    while (tParameters.size() > 7) {
        tPathList.push_back(tParameters.front());
        tParameters.pop_front();
    }

    coremsg(eNormal) << "...initialization finished" << eom;

    KROOTWindow tWindow;
    tWindow.SetName("KGeoBag ROOT Geometry Viewer");

    KGROOTGeometryPainter tPainter;
    tPainter.SetName("ROOT GeometryPainter");
    tPainter.SetDisplayMode(true);
    tPainter.SetWriteMode(true);

    if (tParameters.size() >= 3) {
        string tNormalX(tParameters[0]);
        string tNormalY(tParameters[1]);
        string tNormalZ(tParameters[2]);
        string tSpaceString(" ");
        string tCombine = tNormalX + tSpaceString + tNormalY + tSpaceString + tNormalZ;
        istringstream Converter(tCombine);
        KThreeVector tPlaneNormal;
        Converter >> tPlaneNormal;
        tPainter.SetPlaneNormal(tPlaneNormal);
    }

    if (tParameters.size() >= 6) {
        string tX(tParameters[3]);
        string tY(tParameters[4]);
        string tZ(tParameters[5]);
        string tSpaceString(" ");
        string tCombine = tX + tSpaceString + tY + tSpaceString + tZ;
        istringstream Converter(tCombine);
        KThreeVector tPlanePoint;
        Converter >> tPlanePoint;
        tPainter.SetPlanePoint(tPlanePoint);
    }

    if (tParameters.size() >= 7) {
        if (tParameters[6] == string("true")) {
            tPainter.SetSwapAxis(true);
        }
        else if (tParameters[6] == string("false")) {
            tPainter.SetSwapAxis(false);
        }
    }

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
