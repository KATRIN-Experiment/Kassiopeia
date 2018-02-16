#ifndef KGELECTRODEBUILDER_HH_
#define KGELECTRODEBUILDER_HH_

#include "KGBEM.hh"

#include "KTagged.h"

namespace KGeoBag
{

    template< class BasisPolicy, class BoundaryPolicy >
    class KGBEMAttributor;

    template< class BasisPolicy >
    class KGBEMAttributor< BasisPolicy, KDirichletBoundary > :
        public KTagged,
        public KGBEMData< BasisPolicy, KDirichletBoundary >
    {
        public:
            KGBEMAttributor() :
                    fSurfaces()
            {
            }
            virtual ~KGBEMAttributor()
            {
                KGExtendedSurface< KGBEM< BasisPolicy, KDirichletBoundary > >* tBEMSurface;
                for( std::vector< KGSurface* >::iterator tIt = fSurfaces.begin(); tIt != fSurfaces.end(); tIt++ )
                {
                    tBEMSurface = (*tIt)->template MakeExtension< KGBEM< BasisPolicy, KDirichletBoundary > >();
                    tBEMSurface->SetName( this->GetName() );
                    tBEMSurface->SetTags( this->GetTags() );
                    tBEMSurface->SetBoundaryValue( this->GetBoundaryValue() );
                }
                KGExtendedSpace< KGBEM< BasisPolicy, KDirichletBoundary > >* tBEMSpace;
                for( std::vector< KGSpace* >::iterator tIt = fSpaces.begin(); tIt != fSpaces.end(); tIt++ )
                {
                    tBEMSpace = (*tIt)->template MakeExtension< KGBEM< BasisPolicy, KDirichletBoundary > >();
                    tBEMSpace->SetName( this->GetName() );
                    tBEMSpace->SetTags( this->GetTags() );
                    tBEMSpace->SetBoundaryValue( this->GetBoundaryValue() );
                }
            }

        public:
            void AddSurface( KGSurface* aSurface )
            {
                fSurfaces.push_back( aSurface );
            }
            void AddSpace( KGSpace* aSpace )
            {
                fSpaces.push_back( aSpace );
            }

        private:
            std::vector< KGSurface* > fSurfaces;
            std::vector< KGSpace* > fSpaces;
    };

    template< class BasisPolicy >
    class KGBEMAttributor< BasisPolicy, KNeumannBoundary > :
        public KTagged,
        public KGBEMData< BasisPolicy, KNeumannBoundary >
    {
        public:
            KGBEMAttributor() :
                    fSurfaces()
            {
            }
            virtual ~KGBEMAttributor()
            {
                KGExtendedSurface< KGBEM< BasisPolicy, KNeumannBoundary > >* tBEMSurface;
                for( std::vector< KGSurface* >::iterator tIt = fSurfaces.begin(); tIt != fSurfaces.end(); tIt++ )
                {
                    tBEMSurface = (*tIt)->template MakeExtension< KGBEM< BasisPolicy, KNeumannBoundary > >();
                    tBEMSurface->SetName( this->GetName() );
                    tBEMSurface->SetTags( this->GetTags() );
                    tBEMSurface->SetNormalBoundaryFlux( this->GetNormalBoundaryFlux() );
                }
                KGExtendedSpace< KGBEM< BasisPolicy, KNeumannBoundary > >* tBEMSpace;
                for( std::vector< KGSpace* >::iterator tIt = fSpaces.begin(); tIt != fSpaces.end(); tIt++ )
                {
                    tBEMSpace = (*tIt)->template MakeExtension< KGBEM< BasisPolicy, KNeumannBoundary > >();
                    tBEMSpace->SetName( this->GetName() );
                    tBEMSpace->SetTags( this->GetTags() );
                    tBEMSpace->SetNormalBoundaryFlux( this->GetNormalBoundaryFlux() );
                }
            }

        public:
            void AddSurface( KGSurface* aSurface )
            {
                fSurfaces.push_back( aSurface );
            }
            void AddSpace( KGSpace* aSpace )
            {
                fSpaces.push_back( aSpace );
            }

        private:
            std::vector< KGSurface* > fSurfaces;
            std::vector< KGSpace* > fSpaces;
    };

    typedef KGBEMAttributor< KElectrostaticBasis, KDirichletBoundary > KGElectrostaticDirichletAttributor;
    typedef KGBEMAttributor< KElectrostaticBasis, KNeumannBoundary > KGElectrostaticNeumannAttributor;
    typedef KGBEMAttributor< KMagnetostaticBasis, KDirichletBoundary > KGMagnetostaticDirichletAttributor;
    typedef KGBEMAttributor< KMagnetostaticBasis, KNeumannBoundary > KGMagnetostaticNeumannAttributor;

}

#include "KComplexElement.hh"

namespace katrin
{

    typedef KComplexElement< KGeoBag::KGElectrostaticDirichletAttributor > KGElectrostaticDirichletBuilder;

    template< >
    inline bool KGElectrostaticDirichletBuilder::AddAttribute( KContainer* aContainer )
    {
        using namespace KGeoBag;
        using namespace std;

        if( aContainer->GetName() == "name" )
        {
            fObject->SetName( aContainer->AsReference< string >() );
            return true;
        }
        if( aContainer->GetName() == "value" )
        {
            fObject->SetBoundaryValue( aContainer->AsReference< double >() );
            return true;
        }
        if( aContainer->GetName() == "surfaces" )
        {
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
            vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< string >() );
            vector< KGSpace* >::const_iterator tSpaceIt;
            KGSpace* tSpace;
            const vector< KGSurface* >* tSurfaces;
            vector< KGSurface* >::const_iterator tSurfaceIt;
            KGSurface* tSurface;

            if( tSpaces.size() == 0 )
            {
                coremsg( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return true;
            }

            for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
            {
                tSpace = *tSpaceIt;
                tSurfaces = tSpace->GetBoundaries();
                for( tSurfaceIt = tSurfaces->begin(); tSurfaceIt != tSurfaces->end(); tSurfaceIt++ )
                {
                    tSurface = *tSurfaceIt;
                    fObject->AddSurface( tSurface );
                }
            }
            return true;
        }
        return false;
    }

    typedef KComplexElement< KGeoBag::KGElectrostaticNeumannAttributor > KGElectrostaticNeumannBuilder;

    template< >
    inline bool KGElectrostaticNeumannBuilder::AddAttribute( KContainer* aContainer )
    {
        using namespace KGeoBag;
        using namespace std;

        if( aContainer->GetName() == "name" )
        {
            fObject->SetName( aContainer->AsReference< string >() );
            return true;
        }
        if( aContainer->GetName() == "flux" )
        {
            fObject->SetNormalBoundaryFlux( aContainer->AsReference< double >() );
            return true;
        }
        if( aContainer->GetName() == "surfaces" )
        {
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
            vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< string >() );
            vector< KGSpace* >::const_iterator tSpaceIt;
            KGSpace* tSpace;
            const vector< KGSurface* >* tSurfaces;
            vector< KGSurface* >::const_iterator tSurfaceIt;
            KGSurface* tSurface;

            if( tSpaces.size() == 0 )
            {
                coremsg( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
            {
                tSpace = *tSpaceIt;
                tSurfaces = tSpace->GetBoundaries();
                for( tSurfaceIt = tSurfaces->begin(); tSurfaceIt != tSurfaces->end(); tSurfaceIt++ )
                {
                    tSurface = *tSurfaceIt;
                    fObject->AddSurface( tSurface );
                }
            }
            return true;
        }
        return false;
    }

}

#endif
