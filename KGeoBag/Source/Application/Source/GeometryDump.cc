#include "KGCoreMessage.hh"
#include "KGGeometryPrinter.hh"
#include "KGInterfaceBuilder.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3) {
        cout << "usage: ./GeometryPrinter [--xml] [--json] <config_file_name.xml> <geometry_path> [...]" << endl;
        return -1;
    }

    coremsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv);

    auto& tArgs = tXML.GetArguments();
    bool tUseColors = !tArgs["--colors"].IsVoid();
    bool tWriteJSON = !tArgs["--json"].IsVoid();
    bool tWriteXML = !tArgs["--xml"].IsVoid();

    deque<string> tPathList = tXML.GetArguments().ParameterList();
    tPathList.pop_front();  // strip off config file name

    coremsg(eNormal) << "...initialization finished" << eom;

    KGGeometryPrinter tPainter;
    tPainter.SetName("GeometryPrinter");
    tPainter.SetUseColors(tUseColors);
    tPainter.SetWriteJSON(tWriteJSON);
    tPainter.SetWriteXML(tWriteXML);

    for (auto& tPath : tPathList) {
        for (auto& tSurface : KGInterface::GetInstance()->RetrieveSurfaces(tPath)) {
            tPainter.AddSurface(tSurface);
        }
        for (auto& tSpace : KGInterface::GetInstance()->RetrieveSpaces(tPath)) {
            tPainter.AddSpace(tSpace);
        }
    }

    tPainter.Render();
    tPainter.Write();
    tPainter.Display();

    return 0;
}
