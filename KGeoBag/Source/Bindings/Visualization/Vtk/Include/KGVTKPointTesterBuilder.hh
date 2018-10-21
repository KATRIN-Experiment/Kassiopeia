#ifndef _KGeoBag_KGVTKPointTesterBuilder_hh_
#define _KGeoBag_KGVTKPointTesterBuilder_hh_

#include "KComplexElement.hh"
#include "KGVTKPointTester.hh"
#include "KGVisualizationMessage.hh"

namespace katrin
{

    typedef KComplexElement< KGeoBag::KGVTKPointTester > KGVTKPointTesterBuilder;

    template< >
    inline bool KGVTKPointTesterBuilder::AddAttribute( KContainer* aContainer )
    {
        using namespace std;
        using namespace KGeoBag;

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
                return true;
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
                return true;
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
            aContainer->CopyTo( fObject, &KGVTKPointTester::SetSampleDiskOrigin );
            return true;
        }
        if( aContainer->GetName() == "sample_disk_normal" )
        {
            aContainer->CopyTo( fObject, &KGVTKPointTester::SetSampleDiskNormal );
            return true;
        }
        if( aContainer->GetName() == "sample_disk_radius" )
        {
            aContainer->CopyTo( fObject, &KGVTKPointTester::SetSampleDiskRadius );
            return true;
        }
        if( aContainer->GetName() == "sample_count" )
        {
            aContainer->CopyTo( fObject, &KGVTKPointTester::SetSampleCount );
            return true;
        }
        if( aContainer->GetName() == "sample_color" )
        {
            aContainer->CopyTo( fObject, &KGVTKPointTester::SetSampleColor );
            return true;
        }
        if( aContainer->GetName() == "point_color" )
        {
            aContainer->CopyTo( fObject, &KGVTKPointTester::SetPointColor );
            return true;
        }
        if( aContainer->GetName() == "vertex_size" )
        {
            aContainer->CopyTo( fObject, &KGVTKPointTester::SetVertexSize );
            return true;
        }
        if( aContainer->GetName() == "line_size" )
        {
            aContainer->CopyTo( fObject, &KGVTKPointTester::SetLineSize );
            return true;
        }

        return false;
    }

}

#endif
