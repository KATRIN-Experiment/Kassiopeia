#ifndef KGAXIALMESHBUILDER_HH_
#define KGAXIALMESHBUILDER_HH_

#include "KGAxialMesh.hh"

namespace KGeoBag
{

    class KGAxialMeshAttributor :
            public KTagged,
            public KGAxialMeshData
    {
        public:
            KGAxialMeshAttributor();
            virtual ~KGAxialMeshAttributor();

        public:
            void AddSurface( KGSurface* aSurface );
            void AddSpace( KGSpace* aSpace );

        private:
            vector< KGSurface* > fSurfaces;
            vector< KGSpace* > fSpaces;
    };

}

#include "KComplexElement.hh"

using namespace KGeoBag;
namespace katrin
{

    typedef KComplexElement< KGAxialMeshAttributor > KGAxialMeshBuilder;

    template< >
    inline bool KGAxialMeshBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            fObject->SetName( aContainer->AsReference< string >() );
            return true;
        }
        if( aContainer->GetName() == "surfaces" )
        {
            vector< KGSurface* > tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces( aContainer->AsReference< string >() );
            vector< KGSurface* >::iterator tSurfaceIt;
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
            vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< string >() );
            vector< KGSpace* >::iterator tSpaceIt;
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
        return false;
    }

}

#endif
