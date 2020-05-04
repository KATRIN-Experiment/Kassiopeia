#ifndef KSROOTMAGFIELDPAINTERBUILDER_H
#define KSROOTMAGFIELDPAINTERBUILDER_H

#include "KComplexElement.hh"
#include "KSROOTMagFieldPainter.h"

using namespace Kassiopeia;
namespace katrin
{
typedef KComplexElement<KSROOTMagFieldPainter> KSROOTMagFieldPainterBuilder;

template<> inline bool KSROOTMagFieldPainterBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "x_axis") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetXAxis);
        return true;
    }
    if (aContainer->GetName() == "y_axis") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetYAxis);
        return true;
    }
    if (aContainer->GetName() == "magnetic_field") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetMagneticFieldName);
        return true;
    }
    if (aContainer->GetName() == "r_max") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetRmax);
        return true;
    }
    if (aContainer->GetName() == "z_min") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetZmin);
        return true;
    }
    if (aContainer->GetName() == "z_max") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetZmax);
        return true;
    }
    if (aContainer->GetName() == "z_fix") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetZfix);
        return true;
    }
    if (aContainer->GetName() == "r_steps") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetRsteps);
        return true;
    }
    if (aContainer->GetName() == "z_steps") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetZsteps);
        return true;
    }
    if (aContainer->GetName() == "plot") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetPlot);
        return true;
    }
    if (aContainer->GetName() == "z_axis_logscale") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetUseLogZ);
        return true;
    }
    if (aContainer->GetName() == "magnetic_gradient_numerical") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetGradNumerical);
        return true;
    }
    if (aContainer->GetName() == "draw") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetDraw);
        return true;
    }
    if (aContainer->GetName() == "axial_symmetry") {
        aContainer->CopyTo(fObject, &KSROOTMagFieldPainter::SetAxialSymmetry);
        return true;
    }

    return false;
}

}  // namespace katrin

#endif  // KSROOTMAGFIELDPAINTERBUILDER_H
