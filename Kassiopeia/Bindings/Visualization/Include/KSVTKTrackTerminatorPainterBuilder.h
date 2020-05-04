#ifndef Kassiopeia_KSVTKTrackTerminatorPainterBuilder_h_
#define Kassiopeia_KSVTKTrackTerminatorPainterBuilder_h_

#include "KComplexElement.hh"
#include "KSVTKTrackTerminatorPainter.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSVTKTrackTerminatorPainter> KSVTKTrackTerminatorPainterBuilder;

template<> inline bool KSVTKTrackTerminatorPainterBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::SetFile);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::SetPath);
        return true;
    }
    if (aContainer->GetName() == "outfile") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::SetOutFile);
        return true;
    }
    if (aContainer->GetName() == "point_object") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::SetPointObject);
        return true;
    }
    if (aContainer->GetName() == "point_variable") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::SetPointVariable);
        return true;
    }
    if (aContainer->GetName() == "terminator_object") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::SetTerminatorObject);
        return true;
    }
    if (aContainer->GetName() == "terminator_variable") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::SetTerminatorVariable);
        return true;
    }
    if (aContainer->GetName() == "point_size") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::SetPointSize);
        return true;
    }
    if (aContainer->GetName() == "add_terminator") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::AddTerminator);
        return true;
    }
    if (aContainer->GetName() == "add_color") {
        aContainer->CopyTo(fObject, &KSVTKTrackTerminatorPainter::AddColor);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
