#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

void PrintSurface(KGSurface* tSurface, string indent = "")
{
    KThreeVector origin = tSurface->GetOrigin();
    KThreeVector point = tSurface->Point(origin);
    KThreeVector normal = tSurface->Normal(point);

    cout << indent << "surface " << *tSurface << " in parent <"
         << (tSurface->GetParent() ? tSurface->GetParent()->GetName() : "") << "> :"
         << " origin=" << origin << " point=" << point << " normal=" << normal << endl;
}

void PrintSpace(KGSpace* tSpace, string indent = "")
{
    KThreeVector origin = tSpace->GetOrigin();
    KThreeVector point = tSpace->Point(origin);
    KThreeVector normal = tSpace->Normal(point);

    cout << indent << "space " << *tSpace << " in parent <"
         << (tSpace->GetParent() ? tSpace->GetParent()->GetName() : "") << "> :"
         << " origin=" << origin << " point=" << point << " normal=" << normal << endl;

    for (auto& tChildSurface : *tSpace->GetChildSurfaces())
        PrintSurface(tChildSurface, indent + "  ");

    for (auto& tChildSpace : *tSpace->GetChildSpaces())
        PrintSpace(tChildSpace, indent + "  ");
}


int main(int argc, char** argv)
{
    if (argc < 3) {
        cout << "usage: ./GeometryPrinter <config_file_name.xml> <geometry_path> [...]" << endl;
        return -1;
    }

    coremsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv);

    deque<string> tPathList = tXML.GetArguments().ParameterList();
    tPathList.pop_front();  // strip off config file name

    coremsg(eNormal) << "...initialization finished" << eom;

    for (auto& tPath : tPathList) {
        for (auto& tSurface : KGInterface::GetInstance()->RetrieveSurfaces(tPath)) {
            PrintSurface(tSurface);
        }
        for (auto& tSpace : KGInterface::GetInstance()->RetrieveSpaces(tPath)) {
            PrintSpace(tSpace);
        }
    }

    return 0;
}
