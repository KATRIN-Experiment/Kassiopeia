#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGVTKGeometryPainter.hh"
#include "KGVTKPointTester.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3) {
        cout << "usage: ./PointTester <config_file_name.xml> <geometry_path> [...]" << endl;
        return -1;
    }

    coremsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv);

    deque<string> tPathList = tXML.GetArguments().ParameterList();
    tPathList.pop_front();  // strip off config file name

    coremsg(eNormal) << "...initialization finished" << eom;

    KVTKWindow tWindow;
    tWindow.SetName("KGeoBag Geometry Viewer");
    tWindow.SetFrameColorRed(0.);
    tWindow.SetFrameColorGreen(0.);
    tWindow.SetFrameColorBlue(0.);
    tWindow.SetDisplayMode(true);
    tWindow.SetWriteMode(true);


    KGVTKGeometryPainter tPainter;
    tPainter.SetName("GeometryViewer");
    tPainter.SetDisplayMode(true);
    tPainter.SetWriteMode(true);

    KGVTKPointTester tTester;
    tTester.SetName("PointTester");
    tTester.SetDisplayMode(true);
    tTester.SetWriteMode(true);
    tTester.SetSampleDiskOrigin(KThreeVector(0., 0., 0.));
    tTester.SetSampleDiskNormal(KThreeVector(0., 0., 1.));
    tTester.SetSampleDiskRadius(1.);
    tTester.SetSampleCount(2500);
    tTester.SetSampleColor(KGRGBColor(255, 0, 0));
    tTester.SetPointColor(KGRGBColor(255, 255, 0));
    tTester.SetVertexSize(0.001);
    tTester.SetLineSize(0.001);

    for (auto& tPath : tPathList) {
        for (auto& tSurface : KGInterface::GetInstance()->RetrieveSurfaces(tPath)) {
            tSurface->AcceptNode(&tPainter);
            tTester.AddSurface(tSurface);
        }
        for (auto& tSpace : KGInterface::GetInstance()->RetrieveSpaces(tPath)) {
            tSpace->AcceptNode(&tPainter);
            tTester.AddSpace(tSpace);
        }
    }

    tWindow.AddPainter(&tPainter);
    tWindow.AddPainter(&tTester);
    tWindow.Render();
    tWindow.Write();
    tWindow.Display();
    tWindow.RemovePainter(&tPainter);
    tWindow.RemovePainter(&tTester);

    return 0;
}
