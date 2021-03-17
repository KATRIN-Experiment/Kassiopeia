#ifndef Kassiopeia_KSROOTTrackPainterBuilder_h_
#define Kassiopeia_KSROOTTrackPainterBuilder_h_

#include "KComplexElement.hh"
#include "KSROOTTrackPainter.h"
#include "KSVisualizationMessage.h"
#include "TColor.h"
#include "TROOT.h"

#include <cstdlib>

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSROOTTrackPainter> KSROOTTrackPainterBuilder;

template<> inline bool KSROOTTrackPainterBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "base") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetBase);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetPath);
        return true;
    }
    if (aContainer->GetName() == "plane_normal") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetPlaneNormal);
        return true;
    }
    if (aContainer->GetName() == "plane_point") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetPlanePoint);
        return true;
    }
    if (aContainer->GetName() == "swap_axis") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetSwapAxis);
        return true;
    }
    if (aContainer->GetName() == "epsilon") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetEpsilon);
        return true;
    }
    if (aContainer->GetName() == "x_axis") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetXAxis);
        return true;
    }
    if (aContainer->GetName() == "y_axis") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetYAxis);
        return true;
    }
    if (aContainer->GetName() == "step_output_group_name") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetStepOutputGroupName);
        return true;
    }
    if (aContainer->GetName() == "position_name") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetPositionName);
        return true;
    }
    if (aContainer->GetName() == "track_output_group_name") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetTrackOutputGroupName);
        return true;
    }
    if (aContainer->GetName() == "color_variable") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetColorVariable);
        return true;
    }
    if (aContainer->GetName() == "color_mode") {
        if (aContainer->AsString() == "step") {
            fObject->SetColorMode(KSROOTTrackPainter::eColorStep);
            return true;
        }
        if (aContainer->AsString() == "track") {
            fObject->SetColorMode(KSROOTTrackPainter::eColorTrack);
            return true;
        }
        if (aContainer->AsString() == "fix") {
            fObject->SetColorMode(KSROOTTrackPainter::eColorFix);
            return true;
        }
        if (aContainer->AsString() == "fpd_rings") {
            vismsg(eWarning) << "Backward compatibility warning: " << ret;
            vismsg(eWarning)
                << "To use the fpd color option please use the attribute color_palette instead of color_mode from now on"
                << ret;
            vismsg(eWarning) << "This warning will be be removed in the next version" << eom;
            fObject->SetColorPalette(KSROOTTrackPainter::eColorFPDRings);
            return true;
        }
        return false;
    }
    if (aContainer->GetName() == "color_palette") {
        if (aContainer->AsString() == "default") {
            fObject->SetColorPalette(KSROOTTrackPainter::eColorDefault);
            return true;
        }
        if (aContainer->AsString() == "fpd_rings") {
            fObject->SetColorPalette(KSROOTTrackPainter::eColorFPDRings);
            return true;
        }
        if (aContainer->AsString() == "custom") {
            fObject->SetColorPalette(KSROOTTrackPainter::eColorCustom);
            return true;
        }
        return false;
    }
    if (aContainer->GetName() == "add_color" ||
        aContainer->GetName() == "color")  //color is still used for backward compatibility
    {
        //first find the fraction number, is there is any
        size_t tPos = aContainer->AsString().find(",");
        std::string tColor = aContainer->AsString().substr(0, tPos);
        double tFraction = -1.0;

        if (tPos == std::string::npos) {
            tFraction = -1.0;
        }
        else {
            std::string tFractionString = aContainer->AsString().substr(tPos + 1, std::string::npos);
            tFraction = std::strtod(tFractionString.c_str(), nullptr);
        }

        if (tColor == "kWhite") {
            TColor tColor = *(gROOT->GetColor(kWhite));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kGray") {
            TColor tColor = *(gROOT->GetColor(kGray));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kBlack") {
            TColor tColor = *(gROOT->GetColor(kBlack));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kRed") {
            TColor tColor = *(gROOT->GetColor(kRed));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kGreen") {
            TColor tColor = *(gROOT->GetColor(kGreen));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kBlue") {
            TColor tColor = *(gROOT->GetColor(kBlue));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kYellow") {
            TColor tColor = *(gROOT->GetColor(kYellow));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kMagenta") {
            TColor tColor = *(gROOT->GetColor(kMagenta));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kCyan") {
            TColor tColor = *(gROOT->GetColor(kCyan));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kOrange") {
            TColor tColor = *(gROOT->GetColor(kOrange));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kSpring") {
            TColor tColor = *(gROOT->GetColor(kSpring));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kTeal") {
            TColor tColor = *(gROOT->GetColor(kTeal));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kAzure") {
            TColor tColor = *(gROOT->GetColor(kAzure));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kViolet") {
            TColor tColor = *(gROOT->GetColor(kViolet));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        if (tColor == "kPink") {
            TColor tColor = *(gROOT->GetColor(kPink));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }

        //if color is in hex format
        if (tColor.substr(0, 1) == "#") {
            TColor tColor;
            int tColorNumber = tColor.GetColor(aContainer->AsString().c_str());
            tColor = *(gROOT->GetColor(tColorNumber));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }

        //if color is a digit
        if (isdigit(tColor[0])) {
            int tColorNumber = std::strtol(aContainer->AsString().c_str(), nullptr, 0);
            TColor tColor = *(gROOT->GetColor(tColorNumber));
            fObject->AddBaseColor(tColor, tFraction);
            return true;
        }
        vismsg(eWarning) << "invalid color value: " << aContainer->AsString() << eom;
        return false;
    }
    if (aContainer->GetName() == "draw_options") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetDrawOptions);
        return true;
    }
    if (aContainer->GetName() == "plot_mode") {
        if (aContainer->AsString() == "step") {
            fObject->SetPlotMode(KSROOTTrackPainter::ePlotStep);
            return true;
        }
        if (aContainer->AsString() == "track") {
            fObject->SetPlotMode(KSROOTTrackPainter::ePlotTrack);
            return true;
        }
        return false;
    }
    if (aContainer->GetName() == "axial_mirror") {
        aContainer->CopyTo(fObject, &KSROOTTrackPainter::SetAxialMirror);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
