#include "KGBEMConverter.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"

#include "KGAxialMeshLoop.hh"
#include "KGAxialMeshRing.hh"

#include <cstddef>

#include "KGCoreMessage.hh"

namespace KGeoBag
{

    KGBEMConverter::KGBEMConverter() :
            fSurfaceContainer( NULL ),
            fMinimumArea( 0. ),
            fMaximumAspectRatio(1e100),
            fVerbosity(0),
            fOrigin( KThreeVector::sZero ),
            fXAxis( KThreeVector::sXUnit ),
            fYAxis( KThreeVector::sYUnit ),
            fZAxis( KThreeVector::sZUnit ),
            fAxis(),
            fCurrentOrigin( KThreeVector::sZero ),
            fCurrentXAxis( KThreeVector::sXUnit ),
            fCurrentYAxis( KThreeVector::sYUnit ),
            fCurrentZAxis( KThreeVector::sZUnit ),
            fCurrentAxis()
    {
    }
    KGBEMConverter::~KGBEMConverter()
    {
        Clear();
    }

    void KGBEMConverter::Clear()
    {
        //cout << "clearing content" << endl;

        for( std::vector< Triangle* >::iterator tTriangleIt = fTriangles.begin(); tTriangleIt != fTriangles.end(); ++tTriangleIt )
        {
            delete *tTriangleIt;
        }
        fTriangles.clear();

        for( std::vector< Rectangle* >::iterator tRectangleIt = fRectangles.begin(); tRectangleIt != fRectangles.end(); ++tRectangleIt )
        {
            delete *tRectangleIt;
        }
        fRectangles.clear();

        for( std::vector< LineSegment* >::iterator tLineSegmentIt = fLineSegments.begin(); tLineSegmentIt != fLineSegments.end(); ++tLineSegmentIt )
        {
            delete *tLineSegmentIt;
        }
        fLineSegments.clear();

        for( std::vector< ConicSection* >::iterator tConicSectionIt = fConicSections.begin(); tConicSectionIt != fConicSections.end(); ++tConicSectionIt )
        {
            delete *tConicSectionIt;
        }
        fConicSections.clear();

        for( std::vector< Ring* >::iterator tRingIt = fRings.begin(); tRingIt != fRings.end(); ++tRingIt )
        {
            delete *tRingIt;
        }
        fRings.clear();

        for( std::vector< SymmetricTriangle* >::iterator tTriangleIt = fSymmetricTriangles.begin(); tTriangleIt != fSymmetricTriangles.end(); ++tTriangleIt )
        {
            delete *tTriangleIt;
        }
        fSymmetricTriangles.clear();

        for( std::vector< SymmetricRectangle* >::iterator tRectangleIt = fSymmetricRectangles.begin(); tRectangleIt != fSymmetricRectangles.end(); ++tRectangleIt )
        {
            delete *tRectangleIt;
        }
        fSymmetricRectangles.clear();

        for( std::vector< SymmetricLineSegment* >::iterator tLineSegmentIt = fSymmetricLineSegments.begin(); tLineSegmentIt != fSymmetricLineSegments.end(); ++tLineSegmentIt )
        {
            delete *tLineSegmentIt;
        }
        fSymmetricLineSegments.clear();

        for( std::vector< SymmetricConicSection* >::iterator tConicSectionIt = fSymmetricConicSections.begin(); tConicSectionIt != fSymmetricConicSections.end(); ++tConicSectionIt )
        {
            delete *tConicSectionIt;
        }
        fSymmetricConicSections.clear();

        for( std::vector< SymmetricRing* >::iterator tRingIt = fSymmetricRings.begin(); tRingIt != fSymmetricRings.end(); ++tRingIt )
        {
            delete *tRingIt;
        }
        fSymmetricRings.clear();

        return;
    }

    void KGBEMConverter::SetSystem( const KThreeVector& anOrigin, const KThreeVector& anXAxis, const KThreeVector& aYAxis, const KThreeVector& aZAxis )
    {
        fOrigin = anOrigin;
        fXAxis = anXAxis;
        fYAxis = aYAxis;
        fZAxis = aZAxis;
        fAxis.SetPoints( anOrigin, anOrigin + fZAxis );
        return;

    }
    const KThreeVector& KGBEMConverter::GetOrigin() const
    {
        return fOrigin;
    }
    const KThreeVector& KGBEMConverter::GetXAxis() const
    {
        return fXAxis;
    }
    const KThreeVector& KGBEMConverter::GetYAxis() const
    {
        return fYAxis;
    }
    const KThreeVector& KGBEMConverter::GetZAxis() const
    {
        return fZAxis;
    }
    const KAxis& KGBEMConverter::GetAxis() const
    {
        return fAxis;
    }

    KThreeVector KGBEMConverter::GlobalToInternalPosition( const KThreeVector& aVector )
    {
        KThreeVector tPosition( aVector - fOrigin );
        return KThreeVector( tPosition.Dot( fXAxis ), tPosition.Dot( fYAxis ), tPosition.Dot( fZAxis ) );
    }
    KThreeVector KGBEMConverter::GlobalToInternalVector( const KThreeVector& aVector )
    {
        KThreeVector tVector( aVector );
        return KThreeVector( tVector.Dot( fXAxis ), tVector.Dot( fYAxis ), tVector.Dot( fZAxis ) );
    }
    KThreeVector KGBEMConverter::InternalToGlobalPosition( const KThreeVector& aVector )
    {
        KThreeVector tPosition( aVector.X(), aVector.Y(), aVector.Z() );
        return KThreeVector( fOrigin + tPosition.X() * fXAxis + tPosition.Y() * fYAxis + tPosition.Z() * fZAxis );
    }
    KThreeVector KGBEMConverter::InternalToGlobalVector( const KThreeVector& aVector )
    {
        KThreeVector tVector( aVector.X(), aVector.Y(), aVector.Z() );
        return KThreeVector( tVector.X() * fXAxis + tVector.Y() * fYAxis + tVector.Z() * fZAxis );
    }

    void KGBEMConverter::VisitSurface( KGSurface* aSurface )
    {
        Clear();

        //cout << "visiting surface <" << aSurface->GetName() << ">..." << endl;

        fCurrentOrigin = aSurface->GetOrigin();
        fCurrentXAxis = aSurface->GetXAxis();
        fCurrentYAxis = aSurface->GetYAxis();
        fCurrentZAxis = aSurface->GetZAxis();
        fCurrentAxis.SetPoints( fCurrentOrigin, fCurrentOrigin + fCurrentZAxis );

        DispatchSurface( aSurface );

        return;
    }
    void KGBEMConverter::VisitSpace( KGSpace* aSpace )
    {
        Clear();

        //cout << "visiting space <" << aSpace->GetName() << ">..." << endl;

        fCurrentOrigin = aSpace->GetOrigin();
        fCurrentXAxis = aSpace->GetXAxis();
        fCurrentYAxis = aSpace->GetYAxis();
        fCurrentZAxis = aSpace->GetZAxis();
        fCurrentAxis.SetPoints( fCurrentOrigin, fCurrentOrigin + fCurrentZAxis );

        DispatchSpace( aSpace );

        return;
    }

    KPosition KGBEMConverter::LocalToInternal( const KThreeVector& aVector )
    {
        KThreeVector tGlobalVector( fCurrentOrigin + aVector.X() * fCurrentXAxis + aVector.Y() * fCurrentYAxis + aVector.Z() * fCurrentZAxis );
        KThreeVector tInternalVector( (tGlobalVector - fOrigin).Dot( fXAxis ), (tGlobalVector - fOrigin).Dot( fYAxis ), (tGlobalVector - fOrigin).Dot( fZAxis ) );
        return KPosition( tInternalVector.X(), tInternalVector.Y(), tInternalVector.Z() );
    }
    KPosition KGBEMConverter::LocalToInternal( const KTwoVector& aVector )
    {
        KThreeVector tGlobalVector = fCurrentOrigin + fCurrentZAxis * aVector.Z();
        KTwoVector tInternalVector( (tGlobalVector - fOrigin).Dot( fZAxis ), aVector.R() );
        return KPosition( tInternalVector.R(), 0., tInternalVector.Z() );
    }

    KGBEMMeshConverter::KGBEMMeshConverter()
    {
    }
    KGBEMMeshConverter::KGBEMMeshConverter( KSurfaceContainer& aContainer )
    {
        fSurfaceContainer = &aContainer;
    }
    KGBEMMeshConverter::~KGBEMMeshConverter()
    {
    }

    void KGBEMMeshConverter::DispatchSurface( KGSurface* aSurface )
    {
        Add( aSurface->AsExtension< KGMesh >() );
        return;
    }
    void KGBEMMeshConverter::DispatchSpace( KGSpace* aSpace )
    {
        Add( aSpace->AsExtension< KGMesh >() );
        return;
    }

    void KGBEMMeshConverter::Add( KGMeshData* aData )
    {
        KGMeshElement* tMeshElement;
        KGMeshTriangle* tMeshTriangle;
        KGMeshRectangle* tMeshRectangle;
        KGMeshWire* tMeshWire;

        Triangle* tTriangle;
        Rectangle* tRectangle;
        LineSegment* tLineSegment;

        if( aData != NULL )
        {
            for( vector< KGMeshElement* >::iterator tElementIt = aData->Elements()->begin(); tElementIt != aData->Elements()->end(); tElementIt++ )
            {
                tMeshElement = *tElementIt;

                tMeshTriangle = dynamic_cast< KGMeshTriangle* >( tMeshElement );
                if( (tMeshTriangle != NULL) && (tMeshTriangle->Area() > fMinimumArea) &&  (tMeshTriangle->Aspect() < fMaximumAspectRatio) )
                {
                    tTriangle = new Triangle();
                    tTriangle->SetValues( LocalToInternal( tMeshTriangle->GetP0() ), LocalToInternal( tMeshTriangle->GetP1() ), LocalToInternal( tMeshTriangle->GetP2() ) );
                    fTriangles.push_back( tTriangle );
                    continue;
                }

                tMeshRectangle = dynamic_cast< KGMeshRectangle* >( tMeshElement );
                if( (tMeshRectangle != NULL) && (tMeshRectangle->Area() > fMinimumArea) && (tMeshRectangle->Aspect() < fMaximumAspectRatio) )
                {
                    tRectangle = new Rectangle();
                    tRectangle->SetValues( LocalToInternal( tMeshRectangle->GetP0() ), LocalToInternal( tMeshRectangle->GetP1() ), LocalToInternal( tMeshRectangle->GetP2() ), LocalToInternal( tMeshRectangle->GetP3() ) );
                    fRectangles.push_back( tRectangle );
                    continue;
                }

                tMeshWire = dynamic_cast< KGMeshWire* >( tMeshElement );
                if( (tMeshWire != NULL) && (tMeshWire->Area() > fMinimumArea) && (tMeshWire->Aspect() < fMaximumAspectRatio))
                {
                    tLineSegment = new LineSegment();
                    tLineSegment->SetValues( LocalToInternal( tMeshWire->GetP0() ), LocalToInternal( tMeshWire->GetP1() ), tMeshWire->GetDiameter() );
                    if (tMeshWire->Aspect() < 1.) {
                        coremsg( eWarning ) << "Attention at line segment at P0=" << (KThreeVector)(tLineSegment->GetP0()) << ": Length < Diameter" << eom;
                        coremsg( eNormal ) << "Wires are discretized too finely for the approximation of linear charge density to hold valid." << ret
                                << "Convergence problems of the Robin Hood charge density solver are expected." << ret
                                << "To avoid invalid elements, reduce mesh count and / or mesh power." << eom;
                    }
                    fLineSegments.push_back( tLineSegment );
                    continue;
                }
            }
        }

        return;
    }

    KGBEMAxialMeshConverter::KGBEMAxialMeshConverter()
    {
    }
    KGBEMAxialMeshConverter::KGBEMAxialMeshConverter( KSurfaceContainer& aContainer )
    {
        fSurfaceContainer = &aContainer;
    }
    KGBEMAxialMeshConverter::~KGBEMAxialMeshConverter()
    {
    }

    void KGBEMAxialMeshConverter::DispatchSurface( KGSurface* aSurface )
    {
        Add( aSurface->AsExtension< KGAxialMesh >() );
        return;
    }
    void KGBEMAxialMeshConverter::DispatchSpace( KGSpace* aSpace )
    {
        Add( aSpace->AsExtension< KGAxialMesh >() );
        return;
    }

    void KGBEMAxialMeshConverter::Add( KGAxialMeshData* aData )
    {
        KGAxialMeshElement* tAxialMeshElement;
        KGAxialMeshLoop* tAxialMeshLoop;
        KGAxialMeshRing* tAxialMeshRing;

        ConicSection* tConicSection;
        Ring* tRing;

        if( aData != NULL )
        {
            //cout << "adding axial mesh surface..." << endl;

            if( fAxis.EqualTo( fCurrentAxis ) == false )
            {
                //cout << "...internal origin is <" << fOrigin << ">" << endl;
                //cout << "...internal z axis is <" << fZAxis << ">" << endl;
                //cout << "...current origin is <" << fCurrentOrigin << ">" << endl;
                //cout << "...current z axis is <" << fCurrentZAxis << ">" << endl;
                //cout << "...axes do not match!" << endl;
                return;
            }

            for( vector< KGAxialMeshElement* >::iterator tElementIt = aData->Elements()->begin(); tElementIt != aData->Elements()->end(); tElementIt++ )
            {
                tAxialMeshElement = *tElementIt;

                tAxialMeshLoop = dynamic_cast< KGAxialMeshLoop* >( tAxialMeshElement );
                if( (tAxialMeshLoop != NULL) && (tAxialMeshLoop->Area() > fMinimumArea) )
                {
                    tConicSection = new ConicSection();
                    tConicSection->SetValues( LocalToInternal( tAxialMeshLoop->GetP0() ), LocalToInternal( tAxialMeshLoop->GetP1() ) );
                    fConicSections.push_back( tConicSection );
                    continue;
                }

                tAxialMeshRing = dynamic_cast< KGAxialMeshRing* >( tAxialMeshElement );
                if( (tAxialMeshRing != NULL) && (tAxialMeshRing->Area() > fMinimumArea) )
                {
                    tRing = new Ring();
                    tRing->SetValues( LocalToInternal( tAxialMeshRing->GetP0() ) );
                    fRings.push_back( tRing );
                    continue;
                }
            }

            //cout << "...added <" << fConicSections.size() + fRings.size() << "> components." << endl;
        }

        return;
    }

    KGBEMDiscreteRotationalMeshConverter::KGBEMDiscreteRotationalMeshConverter()
    {
    }
    KGBEMDiscreteRotationalMeshConverter::KGBEMDiscreteRotationalMeshConverter( KSurfaceContainer& aContainer )
    {
        fSurfaceContainer = &aContainer;
    }
    KGBEMDiscreteRotationalMeshConverter::~KGBEMDiscreteRotationalMeshConverter()
    {
    }

    void KGBEMDiscreteRotationalMeshConverter::DispatchSurface( KGSurface* aSurface )
    {
        Add( aSurface->AsExtension< KGDiscreteRotationalMesh >() );
        return;
    }
    void KGBEMDiscreteRotationalMeshConverter::DispatchSpace( KGSpace* aSpace )
    {
        Add( aSpace->AsExtension< KGDiscreteRotationalMesh >() );
        return;
    }

    void KGBEMDiscreteRotationalMeshConverter::Add( KGDiscreteRotationalMeshData* aData )
    {
        KGDiscreteRotationalMeshElement* tMeshElement;
        KGDiscreteRotationalMeshRectangle* tMeshRectangle;
        KGDiscreteRotationalMeshTriangle* tMeshTriangle;
        KGDiscreteRotationalMeshWire* tMeshWire;

        SymmetricTriangle* tTriangles;
        SymmetricRectangle* tRectangles;
        SymmetricLineSegment* tLineSegments;

        KPosition tCenter;
        KDirection tDirection;

        if( aData != NULL )
        {
            //cout << "adding axial mesh surface..." << endl;

            if( fAxis.EqualTo( fCurrentAxis ) == false )
            {
                // improve the hell out of this
                //cout << "...axes do not match!" << endl;
                return;
            }

            tCenter.SetComponents( fAxis.GetCenter().X(), fAxis.GetCenter().Y(), fAxis.GetCenter().Z() );
            tDirection.SetComponents( fAxis.GetDirection().X(), fAxis.GetDirection().Y(), fAxis.GetDirection().Z() );

            for( vector< KGDiscreteRotationalMeshElement* >::iterator tElementIt = aData->Elements()->begin(); tElementIt != aData->Elements()->end(); tElementIt++ )
            {
                tMeshElement = *tElementIt;

                tMeshTriangle = dynamic_cast< KGDiscreteRotationalMeshTriangle* >( tMeshElement );
                if( (tMeshTriangle != NULL) && (tMeshTriangle->Area() > fMinimumArea) && (tMeshTriangle->Aspect() < fMaximumAspectRatio) )
                {
                    tTriangles = new SymmetricTriangle();
                    tTriangles->NewElement()->SetValues( LocalToInternal( tMeshTriangle->Element().GetP0() ), LocalToInternal( tMeshTriangle->Element().GetP1() ), LocalToInternal( tMeshTriangle->Element().GetP2() ) );
                    tTriangles->AddRotationsAboutAxis( tCenter, tDirection, tMeshTriangle->NumberOfElements() );
                    fSymmetricTriangles.push_back( tTriangles );
                    continue;
                }

                tMeshRectangle = dynamic_cast< KGDiscreteRotationalMeshRectangle* >( tMeshElement );
                if( (tMeshRectangle != NULL) && (tMeshRectangle->Area() > fMinimumArea) &&  (tMeshRectangle->Aspect() < fMaximumAspectRatio) )
                {
                    tRectangles = new SymmetricRectangle();
                    tRectangles->NewElement()->SetValues( LocalToInternal( tMeshRectangle->Element().GetP0() ), LocalToInternal( tMeshRectangle->Element().GetP1() ), LocalToInternal( tMeshRectangle->Element().GetP2() ), LocalToInternal( tMeshRectangle->Element().GetP3() ) );
                    tRectangles->AddRotationsAboutAxis( tCenter, tDirection, tMeshRectangle->NumberOfElements() );
                    fSymmetricRectangles.push_back( tRectangles );
                    continue;
                }

                tMeshWire = dynamic_cast< KGDiscreteRotationalMeshWire* >( tMeshElement );
                if( (tMeshWire != NULL) && (tMeshWire->Area() > fMinimumArea) &&  (tMeshWire->Aspect() < fMaximumAspectRatio) )
                {
                    tLineSegments = new SymmetricLineSegment();
                    tLineSegments->NewElement()->SetValues( LocalToInternal( tMeshWire->Element().GetP0() ), LocalToInternal( tMeshWire->Element().GetP1() ), tMeshWire->Element().GetDiameter() );
                    tLineSegments->AddRotationsAboutAxis( tCenter, tDirection, tMeshWire->NumberOfElements() );
                    fSymmetricLineSegments.push_back( tLineSegments );
                    continue;
                }
            }
        }

        return;
    }
}
