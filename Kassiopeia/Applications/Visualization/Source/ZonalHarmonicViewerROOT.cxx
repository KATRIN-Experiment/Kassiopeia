#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGElectrostaticBoundaryField.hh"
#include "KGStaticElectromagnetField.hh"
#include "KGROOTGeometryPainter.hh"
#include "KSFieldFinder.h"
#include "KSElectricKEMField.h"
#include "KSMagneticKEMField.h"
#include "KSROOTZonalHarmonicsPainter.h"
#include "KROOTWindow.h"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"
#include "KToolbox.h"

using namespace KGeoBag;
using namespace Kassiopeia;
using namespace katrin;
using namespace std;

using KEMField::KGElectrostaticBoundaryField;
using KEMField::KGStaticElectromagnetField;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout
            << "usage: ./ZonalHarmonicViewerROOT <config_file_name.xml> <electric_field_name> <magnetic_field_name> [...] --plane={XZ,YZ,XY} [--point='<plane point>']"
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
        // xy-plane is unsupported for zonal harmonic painter
        else
            coremsg(eError) << "plane definition <" << tOptions["plane"] << "> is not supported" << eom;
    }
    if (tOptions.find("point") != tOptions.end()) {
        istringstream Converter(tOptions["point"]);
        Converter >> tPlanePoint;
    }

    deque<string> tFieldList = tParameters;

    coremsg(eNormal) << "...initialization finished" << eom;

    KROOTWindow tWindow;
    tWindow.SetName("KGeoBag ROOT Geometry Viewer");

    KGROOTGeometryPainter tPainter;
    tPainter.SetName("ROOT GeometryPainter");
    tPainter.SetDisplayMode(true);
    tPainter.SetWriteMode(true);

    tPainter.SetPlaneNormal(tPlaneNormal);
    tPainter.SetPlanePoint(tPlanePoint);
    tPainter.SetSwapAxis(tSwapAxes);

    coremsg(eNormal) << "painting with plane normal " << tPlaneNormal << ", plane point " << tPlanePoint << eom;

    KSROOTZonalHarmonicsPainter tPainterZH;
    tPainterZH.SetName("ROOT ZonalHarmonicsPainter");
    tPainterZH.SetDisplayMode(true);
    tPainterZH.SetWriteMode(true);

    tPainterZH.SetDrawSourcePoints(true);
    tPainterZH.SetDrawConvergenceArea(true);
    tPainterZH.SetDrawExpansionArea(true);

    for (auto& tField : tFieldList) {
        KGElectrostaticBoundaryField* tElectricFieldObject = KToolbox::GetInstance().Get<KGElectrostaticBoundaryField>(tField);
        if (! tElectricFieldObject) {
            // necessary if field is define inside the <kassiopeia> tag
            auto* tKEMFieldObject = KToolbox::GetInstance().Get<KSElectricKEMField>(tField);
            if (tKEMFieldObject)
                tElectricFieldObject = dynamic_cast<KGElectrostaticBoundaryField*>(tKEMFieldObject->GetElectricField());
        }
        if (tElectricFieldObject) {
            coremsg(eNormal) << "adding electric field: " << tField << eom;
            tPainterZH.SetElectricFieldName(tField);

            for (auto& tSurface : tElectricFieldObject->GetSurfaces()) {
                tPainter.AddSurface(tSurface);
            }
            for (auto& tSpace : tElectricFieldObject->GetSpaces()) {
                tPainter.AddSpace(tSpace);
            }
        }

        KGStaticElectromagnetField* tMagneticFieldObject = KToolbox::GetInstance().Get<KGStaticElectromagnetField>(tField);
        if (! tMagneticFieldObject) {
            // necessary if field is define inside the <kassiopeia> tag
            auto* tKEMFieldObject = KToolbox::GetInstance().Get<KSMagneticKEMField>(tField);
            if (tKEMFieldObject)
                tMagneticFieldObject = dynamic_cast<KGStaticElectromagnetField*>(tKEMFieldObject->GetMagneticField());
        }
        if (tMagneticFieldObject) {
            coremsg(eNormal) << "adding magnetic field: " << tField << eom;
            tPainterZH.SetMagneticFieldName(tField);

            for (auto& tSurface : tMagneticFieldObject->GetSurfaces()) {
                tPainter.AddSurface(tSurface);
            }
            for (auto& tSpace : tMagneticFieldObject->GetSpaces()) {
                tPainter.AddSpace(tSpace);
            }
        }
    }

    // set ZH painiting limits based on geometry
    tPainter.Render();  // update bounds
    tPainterZH.SetZmin(tPainter.GetXMin());
    tPainterZH.SetZmax(tPainter.GetXMax());
    tPainterZH.SetRmax(max(fabs(tPainter.GetYMin()), fabs(tPainter.GetYMax())));

    //tPainterZH.SetZdist();
    //tPainterZH.SetRdist();

    tWindow.AddPainter(&tPainter);
    tWindow.AddPainter(&tPainterZH);
    tWindow.Render();
    tWindow.Display();
    tWindow.Write();
    tWindow.RemovePainter(&tPainter);
    tWindow.RemovePainter(&tPainterZH);

    return 0;
}
