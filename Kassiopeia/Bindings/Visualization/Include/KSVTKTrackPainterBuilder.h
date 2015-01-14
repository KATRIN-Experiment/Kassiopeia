#ifndef Kassiopeia_KSVTKTrackPainterBuilder_h_
#define Kassiopeia_KSVTKTrackPainterBuilder_h_

#include "KComplexElement.hh"
#include "KSVTKTrackPainter.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSVTKTrackPainter > KSVTKTrackPainterBuilder;

    template< >
    inline bool KSVTKTrackPainterBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "file" )
        {
            aContainer->CopyTo( fObject, &KSVTKTrackPainter::SetFile );
            return true;
        }
        if( aContainer->GetName() == "path" )
        {
            aContainer->CopyTo( fObject, &KSVTKTrackPainter::SetPath );
            return true;
        }
        if( aContainer->GetName() == "outfile" )
        {
            aContainer->CopyTo( fObject, &KSVTKTrackPainter::SetOutFile );
            return true;
        }
        if( aContainer->GetName() == "point_object" )
        {
            aContainer->CopyTo( fObject, &KSVTKTrackPainter::SetPointObject );
            return true;
        }
        if( aContainer->GetName() == "point_variable" )
        {
            aContainer->CopyTo( fObject, &KSVTKTrackPainter::SetPointVariable );
            return true;
        }
        if( aContainer->GetName() == "color_object" )
        {
            aContainer->CopyTo( fObject, &KSVTKTrackPainter::SetColorObject );
            return true;
        }
        if( aContainer->GetName() == "color_variable" )
        {
            aContainer->CopyTo( fObject, &KSVTKTrackPainter::SetColorVariable );
            return true;
        }
        return false;
    }

}

#endif
