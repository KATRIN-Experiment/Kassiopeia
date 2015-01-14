#ifndef _KGeoBag_KGVTKOutsideTesterBuilder_hh_
#define _KGeoBag_KGVTKOutsideTesterBuilder_hh_

#include "KComplexElement.hh"
#include "KGVTKOutsideTester.hh"
#include "KGVisualizationMessage.hh"

using namespace KGeoBag;
namespace katrin
{

    typedef KComplexElement< KGVTKOutsideTester > KGVTKOutsideTesterBuilder;

    template< >
    inline bool KGVTKOutsideTesterBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "surfaces" )
        {
            if( aContainer->AsReference< string >().size() == 0 )
            {
                return true;
            }

            vector< KGSurface* > tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces( aContainer->AsReference< string >() );
            vector< KGSurface* >::const_iterator tSurfaceIt;
            KGSurface* tSurface;

            if( tSurfaces.size() == 0 )
            {
                coremsg( eWarning ) << "no surfaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
            {
                tSurface = *tSurfaceIt;
                fObject->AddSurface( tSurface );
            }
            return true;
        }
        if( aContainer->GetName() == "spaces" )
        {
            if( aContainer->AsReference< string >().size() == 0 )
            {
                return true;
            }

            vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< string >() );
            vector< KGSpace* >::const_iterator tSpaceIt;
            KGSpace* tSpace;

            if( tSpaces.size() == 0 )
            {
                coremsg( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
            {
                tSpace = *tSpaceIt;
                fObject->AddSpace( tSpace );
            }
            return true;
        }

        if( aContainer->GetName() == "sample_disk_origin" )
        {
            aContainer->CopyTo( fObject, &KGVTKOutsideTester::SetSampleDiskOrigin );
            return true;
        }
        if( aContainer->GetName() == "sample_disk_normal" )
        {
            aContainer->CopyTo( fObject, &KGVTKOutsideTester::SetSampleDiskNormal );
            return true;
        }
        if( aContainer->GetName() == "sample_disk_radius" )
        {
            aContainer->CopyTo( fObject, &KGVTKOutsideTester::SetSampleDiskRadius );
            return true;
        }
        if( aContainer->GetName() == "sample_count" )
        {
            aContainer->CopyTo( fObject, &KGVTKOutsideTester::SetSampleCount );
            return true;
        }
        if( aContainer->GetName() == "inside_color" )
        {
            aContainer->CopyTo( fObject, &KGVTKOutsideTester::SetInsideColor );
            return true;
        }
        if( aContainer->GetName() == "outside_color" )
        {
            aContainer->CopyTo( fObject, &KGVTKOutsideTester::SetOutsideColor );
            return true;
        }
        if( aContainer->GetName() == "vertex_size" )
        {
            aContainer->CopyTo( fObject, &KGVTKOutsideTester::SetVertexSize );
            return true;
        }

        return false;
    }

}

#endif
