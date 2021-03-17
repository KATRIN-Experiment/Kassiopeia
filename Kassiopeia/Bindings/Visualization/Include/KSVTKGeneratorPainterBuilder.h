#ifndef Kassiopeia_KSVTKGeneratorPainterBuilder_h_
#define Kassiopeia_KSVTKGeneratorPainterBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldFinder.h"
#include "KSVTKGeneratorPainter.h"

using namespace Kassiopeia;

namespace katrin
{

typedef KComplexElement<KSVTKGeneratorPainter> KSVTKGeneratorPainterBuilder;

template<> inline bool KSVTKGeneratorPainterBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSVTKGeneratorPainter::SetPath);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KSVTKGeneratorPainter::SetFile);
        return true;
    }
    if (aContainer->GetName() == "electric_field") {
        auto tField = getElectricField(aContainer->AsString());
        if (!tField)
            return false;
        fObject->AddElectricField(tField);
        return true;
    }
    if (aContainer->GetName() == "magnetic_field") {
        auto tField = getMagneticField(aContainer->AsString());
        if (!tField)
            return false;
        fObject->AddMagneticField(tField);
        return true;
    }
    if (aContainer->GetName() == "num_samples") {
        aContainer->CopyTo(fObject, &KSVTKGeneratorPainter::SetNumSamples);
        return true;
    }
    if (aContainer->GetName() == "scale_factor") {
        aContainer->CopyTo(fObject, &KSVTKGeneratorPainter::SetScaleFactor);
        return true;
    }
    if (aContainer->GetName() == "color_variable") {
        aContainer->CopyTo(fObject, &KSVTKGeneratorPainter::SetColorVariable);
        return true;
    }
    if (aContainer->GetName() == "add_generator") {
        aContainer->CopyTo(fObject, &KSVTKGeneratorPainter::AddGenerator);
        return true;
    }
    if (aContainer->GetName() == "add_color") {
        aContainer->CopyTo(fObject, &KSVTKGeneratorPainter::AddColor);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
