#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGVTKGeometryPainter.hh"
#include "KSVTKTrackPainter.h"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"
#include "KGlobals.hh"
#include "KBaseStringUtils.h"

using namespace Kassiopeia;
using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 4) {
        cout << "usage: ./TrackViewer <config_file_name.xml> <output_file.root> <point_group:point_variable[:color_variable]> [geometry_path] [...]" << endl;
        cout << "You need to specify an output group and variable that exists in the output file." << endl
             << "The variable must contain position vector data. A scalar color variable is optional." << endl;
        return -1;
    }

    coremsg(eNormal) << "starting initialization..." << eom;

    KGlobals::GetInstance().SetBatchMode(true);  // make sure to NOT show any ROOT/VTK windows

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv);

    auto tPathList = tXML.GetArguments().ParameterList();
    tPathList.pop_front();  // strip off config file name
    string tFileName = tPathList.at(0);
    tPathList.pop_front();
    string tPointName = tPathList.at(0);
    tPathList.pop_front();

    auto tPos = tPointName.find_first_of(":");
    if (tPos == string::npos) {
        cout << "error: you need to specify a point object and variable, e.g.: 'output_step_world/position'" << endl;
        return -2;
    }
    string tPointObject = tPointName.substr(0, tPos);
    string tPointVariable = tPointName.substr(tPos+1, string::npos);
    string tColorVariable = "";

    tPos = tPointVariable.find_first_of(":");
    if (tPos != string::npos) {
        tColorVariable = tPointVariable.substr(tPos+1, string::npos);
        tPointVariable = tPointVariable.substr(0, tPos);
    }

    tPos = tFileName.find_last_of("/");
    string tFilePath;
    if (tPos != string::npos) {
        tFilePath = tFileName.substr(0, tPos);
        tFileName = tFileName.substr(tPos+1, string::npos);
    }

    coremsg(eNormal) << "...initialization finished" << eom;

    KGlobals::GetInstance().SetBatchMode(false);  // make sure that the viewer window is shown here

    KVTKWindow tWindow;
    tWindow.SetName("Kassiopeia Track Viewer");
    tWindow.SetFrameColorRed(0.);
    tWindow.SetFrameColorGreen(0.);
    tWindow.SetFrameColorBlue(0.);
    tWindow.SetDisplayMode(true);
    tWindow.SetWriteMode(true);

    KGVTKGeometryPainter tPainter;
    tPainter.SetName("GeometryPainter");
    tPainter.SetDisplayMode(true);
    tPainter.SetWriteMode(true);

    for (auto& tPath : tPathList) {
        for (auto& tSurface : KGInterface::GetInstance()->RetrieveSurfaces(tPath)) {
            tPainter.AddSurface(tSurface);
        }
        for (auto& tSpace : KGInterface::GetInstance()->RetrieveSpaces(tPath)) {
            tPainter.AddSpace(tSpace);
        }
    }

    KSVTKTrackPainter tTrackPainter;
    tTrackPainter.SetName("TrackPainter");
    tTrackPainter.SetDisplayMode(true);
    tTrackPainter.SetWriteMode(true);
    tTrackPainter.SetPath(tFilePath);
    tTrackPainter.SetFile(tFileName);
    tTrackPainter.SetPointObject(tPointObject);
    tTrackPainter.SetPointVariable(tPointVariable);
    tTrackPainter.SetColorObject(tPointObject);
    tTrackPainter.SetColorVariable(tColorVariable);

    tWindow.AddPainter(&tPainter);
    tWindow.AddPainter(&tTrackPainter);
    tWindow.Render();
    tWindow.Write();
    tWindow.Display();
    tWindow.RemovePainter(&tPainter);
    tWindow.RemovePainter(&tTrackPainter);

    return 0;
}
