#ifndef KSROOTZONALHARMONICSPAINTERBUILDER_H
#define KSROOTZONALHARMONICSPAINTERBUILDER_H

#include "KComplexElement.hh"
#include "KSROOTZonalHarmonicsPainter.h"
#include "KSVisualizationMessage.h"


using namespace Kassiopeia;
namespace katrin
{
typedef KComplexElement<KSROOTZonalHarmonicsPainter> KSROOTZonalHarmonicsPainterBuilder;

template<> inline bool KSROOTZonalHarmonicsPainterBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "x_axis") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetXAxis);
        return true;
    }
    if (aContainer->GetName() == "y_axis") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetYAxis);
        return true;
    }
    if (aContainer->GetName() == "electric_field") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetElectricFieldName);
        return true;
    }
    if (aContainer->GetName() == "magnetic_field") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetMagneticFieldName);
        return true;
    }
    if (aContainer->GetName() == "r_min") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetRMin);
        return true;
    }
    if (aContainer->GetName() == "r_max") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetRMax);
        return true;
    }
    if (aContainer->GetName() == "z_min") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetZMin);
        return true;
    }
    if (aContainer->GetName() == "z_max") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetZMax);
        return true;
    }
    if (aContainer->GetName() == "z_dist") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetZDist);
        return true;
    }
    if (aContainer->GetName() == "r_dist") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetRDist);
        return true;
    }
    if (aContainer->GetName() == "r_steps") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetRMaxSteps);
        return true;
    }
    if (aContainer->GetName() == "z_steps") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetZMaxSteps);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetPath);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetFile);
        return true;
    }
    if (aContainer->GetName() == "write") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetWriteMode);
        return true;
    }
    if (aContainer->GetName() == "draw_source_points") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetDrawSourcePoints);
        return true;
    }
    if (aContainer->GetName() == "draw_convergence_area") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetDrawCentralBoundary);
        return true;
    }
    if (aContainer->GetName() == "draw_central_boundary") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetDrawCentralBoundary);
        return true;
    }
    if (aContainer->GetName() == "draw_remote_boundary") {
        aContainer->CopyTo(fObject, &KSROOTZonalHarmonicsPainter::SetDrawRemoteBoundary);
        return true;
    }

#if 0
        if( aContainer->GetName() == "geometry_type" )
        {
            if( aContainer->AsReference< std::string >() == "volume" )
            {
                vismsg(eDebug) << "setting painter to volume write mode" << eom;
                aContainer->CopyTo( fObject, &KSROOTZonalHarmonicsPainter::SetGeometryType );
                return true;
            }
            if( aContainer->AsReference< std::string >() == "surface" )
            {
                vismsg(eDebug) << "setting painter to surface write mode" << eom;
                aContainer->CopyTo( fObject, &KSROOTZonalHarmonicsPainter::SetGeometryType );
                return true;
            }
            return false;
        }
        if( aContainer->GetName() == "radial_safety_margin" )
        {
            aContainer->CopyTo( fObject, &KSROOTZonalHarmonicsPainter::SetRadialSafetyMargin );
            return true;
        }
#endif

    return false;
}

}  // namespace katrin

#endif  // KSROOTZONALHARMONICSPAINTERBUILDER_H
