#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGRandomPointGenerator.hh"
#include "KGVTKGeometryPainter.hh"
#include "KGVTKRandomPointTester.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3) {
        cout << "usage: ./NormalTester <config_file_name.xml> <geometry_path> [...] --samples=N" << endl;
        return -1;
    }

    coremsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv);

    unsigned int tSampleCount = tXML.GetArguments().GetOption("samples").Or(100000);
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
    tPainter.SetName("RandomPointViewer");
    tPainter.SetDisplayMode(true);
    tPainter.SetWriteMode(true);

    KGVTKRandomPointTester tTester;
    tTester.SetName("RandomPointTester");
    tTester.SetDisplayMode(true);
    tTester.SetWriteMode(true);
    tTester.SetSampleColor(KGRGBColor(0, 255, 0));
    tTester.SetVertexSize(0.001);

    std::vector<KGSurface*> tSurfaces;
    std::vector<KGSpace*> tSpaces;

    for (auto& tPath : tPathList) {
        for (auto& tSurface : KGInterface::GetInstance()->RetrieveSurfaces(tPath)) {
            tSurface->AcceptNode(&tPainter);
            tTester.AddSurface(tSurface);
            tSurfaces.push_back(tSurface);
        }
        for (auto& tSpace : KGInterface::GetInstance()->RetrieveSpaces(tPath)) {
            tSpace->AcceptNode(&tPainter);
            tTester.AddSpace(tSpace);
            tSpaces.push_back(tSpace);
        }
    }

    coremsg(eNormal) << "starting calculation of points (" << tSampleCount << ")..." << eom;
    KGRandomPointGenerator random;
    vector<KThreeVector*> tPoints;

    for (unsigned int i = 0; i < tSampleCount; ++i) {
        tPoints.push_back(new KThreeVector(random.Random(tSpaces)));
    }
    coremsg(eNormal) << "...calculation of points finished" << eom;

    coremsg(eNormal) << "starting calculation of points per volume..." << eom;
    vector<unsigned int> tCounter;

    for (auto s = tSpaces.begin(); s != tSpaces.end(); ++s) {
        tCounter.push_back(0);
    }

    for (auto& p : tPoints) {
        unsigned int c = 0;
        for (auto s = tSpaces.begin(); s != tSpaces.end(); ++s, ++c) {
            if (!(*s)->Outside(*p)) {
                tCounter[c]++;
                break;
            }
        }
    }
    coremsg(eNormal) << "...calculation of points per volume finished:" << eom;
    unsigned int c = 0;
    unsigned int tTotalPoints = 0;
    for (auto s = tSpaces.begin(); s != tSpaces.end(); ++s, ++c) {
        coremsg(eNormal) << "   <" << (*s)->GetName() << ">: V = " << (*s)->AsExtension<KGMetrics>()->GetVolume()
                         << " m^3; "
                         << "points = " << tCounter[c] << "; "
                         << "density = " << (double(tCounter[c]) / (*s)->AsExtension<KGMetrics>()->GetVolume())
                         << " points / m^3" << eom;

        tTotalPoints += tCounter[c];
    }
    coremsg(eNormal) << "   total points = " << tTotalPoints << eom;

    tTester.SetSamplePoints(tPoints);
    tWindow.AddPainter(&tPainter);
    tWindow.AddPainter(&tTester);
    tWindow.Render();
    tWindow.Write();
    tWindow.Display();
    tWindow.RemovePainter(&tPainter);
    tWindow.RemovePainter(&tTester);

    return 0;
}
