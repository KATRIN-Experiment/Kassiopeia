#ifndef KGBEMCONVERTER_DEF
#define KGBEMCONVERTER_DEF

#include "KGBEM.hh"

#include "KGMesh.hh"
#include "KGAxialMesh.hh"
#include "KGDiscreteRotationalMesh.hh"

#include "KThreeVector.hh"
#include "KAxis.hh"

#include <vector>

namespace KGeoBag
{

    template< template< class, class > class XNode, class XListOne, class XListTwo >
    class KGDualHierarchy;

    template< template< class, class > class XNode, class XTypeOne, class XHeadTwo, class XTailTwo >
    class KGDualHierarchy< XNode, XTypeOne, KTypelist< XHeadTwo, XTailTwo > > :
        public KGDualHierarchy< XNode, XTypeOne, XTailTwo >,
        public XNode< XTypeOne, XHeadTwo >
    {
    };

    template< template< class, class > class XNode, class XTypeOne, class XHeadTwo >
    class KGDualHierarchy< XNode, XTypeOne, KTypelist< XHeadTwo, KNullType > > :
        public XNode< XTypeOne, XHeadTwo >
    {
    };

    template< template< class, class > class XNode, class XHeadOne, class XTailOne, class XHeadTwo, class XTailTwo >
    class KGDualHierarchy< XNode, KTypelist< XHeadOne, XTailOne >, KTypelist< XHeadTwo, XTailTwo > > :
        public KGDualHierarchy< XNode, XHeadOne, KTypelist< XHeadTwo, XTailTwo > >,
        public KGDualHierarchy< XNode, XTailOne, KTypelist< XHeadTwo, XTailTwo > >
    {
    };

    template< template< class, class > class XNode, class XHeadOne, class XHeadTwo, class XTailTwo >
    class KGDualHierarchy< XNode, KTypelist< XHeadOne, KNullType >, KTypelist< XHeadTwo, XTailTwo > > :
        public KGDualHierarchy< XNode, XHeadOne, KTypelist< XHeadTwo, XTailTwo > >
    {
    };

}

namespace KGeoBag
{

    class KGBEMConverter :
        public KGVisitor,
        public KGSurface::Visitor,
        public KGSpace::Visitor
    {
        protected:
            KGBEMConverter();

        public:
            virtual ~KGBEMConverter();

        public:
            void SetSurfaceContainer( KSurfaceContainer* aContainer )
            {
                fSurfaceContainer = aContainer;
                return;
            }
            void SetMinimumArea( double aMinimumArea )
            {
                fMinimumArea = aMinimumArea;
                return;
            }

        protected:
            KSurfaceContainer* fSurfaceContainer;
            double fMinimumArea;

            class Triangle :
                public KEMField::KTriangle
            {
                public:
                    typedef KEMField::KTriangle ShapePolicy;

                    Triangle()
                    {
                    }
                    virtual ~Triangle()
                    {
                    }
            };

            class Rectangle :
                public KEMField::KRectangle
            {
                public:
                    typedef KEMField::KRectangle ShapePolicy;

                    Rectangle()
                    {
                    }
                    virtual ~Rectangle()
                    {
                    }
            };

            class LineSegment :
                public KEMField::KLineSegment
            {
                public:
                    typedef KEMField::KLineSegment ShapePolicy;

                    LineSegment()
                    {
                    }
                    virtual ~LineSegment()
                    {
                    }
            };

            class ConicSection :
                public KEMField::KConicSection
            {
                public:
                    typedef KEMField::KConicSection ShapePolicy;

                    ConicSection()
                    {
                    }
                    virtual ~ConicSection()
                    {
                    }
            };

            class Ring :
                public KEMField::KRing
            {
                public:
                    typedef KEMField::KRing ShapePolicy;

                    Ring()
                    {
                    }
                    virtual ~Ring()
                    {
                    }
            };

            class SymmetricTriangle :
                public KEMField::KSymmetryGroup< KEMField::KTriangle >
            {
                public:
                    typedef KEMField::KSymmetryGroup< KEMField::KTriangle > ShapePolicy;

                    SymmetricTriangle()
                    {
                    }
                    virtual ~SymmetricTriangle()
                    {
                    }
            };

            class SymmetricRectangle :
                public KEMField::KSymmetryGroup< KEMField::KRectangle >
            {
                public:
                    typedef KEMField::KSymmetryGroup< KEMField::KRectangle > ShapePolicy;

                    SymmetricRectangle()
                    {
                    }
                    virtual ~SymmetricRectangle()
                    {
                    }
            };

            class SymmetricLineSegment :
                public KEMField::KSymmetryGroup< KEMField::KLineSegment >
            {
                public:
                    typedef KEMField::KSymmetryGroup< KEMField::KLineSegment > ShapePolicy;

                    SymmetricLineSegment()
                    {
                    }
                    virtual ~SymmetricLineSegment()
                    {
                    }
            };

            class SymmetricConicSection :
                public KEMField::KSymmetryGroup< KEMField::KConicSection >
            {
                public:
                    typedef KEMField::KSymmetryGroup< KEMField::KConicSection > ShapePolicy;

                    SymmetricConicSection()
                    {
                    }
                    virtual ~SymmetricConicSection()
                    {
                    }
            };

            class SymmetricRing :
                public KEMField::KSymmetryGroup< KEMField::KRing >
            {
                public:
                    typedef KEMField::KSymmetryGroup< KEMField::KRing > ShapePolicy;

                    SymmetricRing()
                    {
                    }
                    virtual ~SymmetricRing()
                    {
                    }
            };

            void Clear();

            std::vector< Triangle* > fTriangles;
            std::vector< Rectangle* > fRectangles;
            std::vector< LineSegment* > fLineSegments;
            std::vector< ConicSection* > fConicSections;
            std::vector< Ring* > fRings;
            std::vector< SymmetricTriangle* > fSymmetricTriangles;
            std::vector< SymmetricRectangle* > fSymmetricRectangles;
            std::vector< SymmetricLineSegment* > fSymmetricLineSegments;
            std::vector< SymmetricConicSection* > fSymmetricConicSections;
            std::vector< SymmetricRing* > fSymmetricRings;

        public:
            void SetSystem( const KThreeVector& anOrigin, const KThreeVector& anXAxis, const KThreeVector& aYAxis, const KThreeVector& aZAxis );
            const KThreeVector& GetOrigin() const;
            const KThreeVector& GetXAxis() const;
            const KThreeVector& GetYAxis() const;
            const KThreeVector& GetZAxis() const;
            const KAxis& GetAxis() const;

            KEMThreeVector GlobalToInternalPosition( const KThreeVector& aPosition );
            KEMThreeVector GlobalToInternalVector( const KThreeVector& aVector );
            KThreeVector InternalToGlobalPosition( const KEMThreeVector& aVector );
            KThreeVector InternalToGlobalVector( const KEMThreeVector& aVector );

            void VisitSurface( KGSurface* aSurface );
            void VisitSpace( KGSpace* aSpace );

        protected:
            KPosition LocalToInternal( const KThreeVector& aVector );
            KPosition LocalToInternal( const KTwoVector& aVector );

            virtual void DispatchSurface( KGSurface* aSurface ) = 0;
            virtual void DispatchSpace( KGSpace* aSpace ) = 0;

        protected:
            KThreeVector fOrigin;
            KThreeVector fXAxis;
            KThreeVector fYAxis;
            KThreeVector fZAxis;
            KAxis fAxis;

            KThreeVector fCurrentOrigin;
            KThreeVector fCurrentXAxis;
            KThreeVector fCurrentYAxis;
            KThreeVector fCurrentZAxis;
            KAxis fCurrentAxis;
    };

    template< class XBasisPolicy, class XBoundaryPolicy >
    class KGBEMConverterNode :
        virtual public KGBEMConverter,
        public KGExtendedSurface< KGBEM< XBasisPolicy, XBoundaryPolicy > >::Visitor,
        public KGExtendedSpace< KGBEM< XBasisPolicy, XBoundaryPolicy > >::Visitor
    {
        public:
            KGBEMConverterNode() :
                    KGBEMConverter()
            {
            }
            virtual ~KGBEMConverterNode()
            {
            }

        public:
            void VisitExtendedSurface( KGExtendedSurface< KGBEM< XBasisPolicy, XBoundaryPolicy > >* aSurface )
            {
                Add( aSurface );
                return;
            }

            void VisitExtendedSpace( KGExtendedSpace< KGBEM< XBasisPolicy, XBoundaryPolicy > >* aSpace )
            {
                Add( aSpace );
                return;
            }

        private:
            void Add( KGBEMData< XBasisPolicy, XBoundaryPolicy >* aBEM )
            {
                //cout << "adding bem surface of type < " << XBasisPolicy::Name() << ", " << XBoundaryPolicy::Name() << " >..." << endl;

                for( vector< Triangle* >::iterator tTriangleIt = fTriangles.begin(); tTriangleIt != fTriangles.end(); tTriangleIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KTriangle >( *aBEM, *aBEM, **tTriangleIt ) );
                }
                for( vector< Rectangle* >::iterator tRectangleIt = fRectangles.begin(); tRectangleIt != fRectangles.end(); tRectangleIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KRectangle >( *aBEM, *aBEM, **tRectangleIt ) );
                }
                for( vector< LineSegment* >::iterator tLineSegmentIt = fLineSegments.begin(); tLineSegmentIt != fLineSegments.end(); tLineSegmentIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KLineSegment >( *aBEM, *aBEM, **tLineSegmentIt ) );
                }
                for( vector< ConicSection* >::iterator tConicSectionIt = fConicSections.begin(); tConicSectionIt != fConicSections.end(); tConicSectionIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KConicSection >( *aBEM, *aBEM, **tConicSectionIt ) );
                }
                for( vector< Ring* >::iterator tRingIt = fRings.begin(); tRingIt != fRings.end(); tRingIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KRing >( *aBEM, *aBEM, **tRingIt ) );
                }
                for( vector< SymmetricTriangle* >::iterator tTriangleIt = fSymmetricTriangles.begin(); tTriangleIt != fSymmetricTriangles.end(); tTriangleIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KSymmetryGroup< KTriangle > >( *aBEM, *aBEM, **tTriangleIt ) );
                }
                for( vector< SymmetricRectangle* >::iterator tRectangleIt = fSymmetricRectangles.begin(); tRectangleIt != fSymmetricRectangles.end(); tRectangleIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KSymmetryGroup< KRectangle > >( *aBEM, *aBEM, **tRectangleIt ) );
                }
                for( vector< SymmetricLineSegment* >::iterator tLineSegmentIt = fSymmetricLineSegments.begin(); tLineSegmentIt != fSymmetricLineSegments.end(); tLineSegmentIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KSymmetryGroup< KLineSegment > >( *aBEM, *aBEM, **tLineSegmentIt ) );
                }
                for( vector< SymmetricConicSection* >::iterator tConicSectionIt = fSymmetricConicSections.begin(); tConicSectionIt != fSymmetricConicSections.end(); tConicSectionIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KSymmetryGroup< KConicSection > >( *aBEM, *aBEM, **tConicSectionIt ) );
                }
                for( vector< SymmetricRing* >::iterator tRingIt = fSymmetricRings.begin(); tRingIt != fSymmetricRings.end(); tRingIt++ )
                {
                    fSurfaceContainer->push_back( new KSurface< XBasisPolicy, XBoundaryPolicy, KSymmetryGroup< KRing > >( *aBEM, *aBEM, **tRingIt ) );
                }

                //cout << "...surface container has <" << fSurfaceContainer->size() << "> elements." << endl;
                return;
            }

    };

    class KGBEMMeshConverter :
        public KGDualHierarchy< KGBEMConverterNode, KBasisTypes, KBoundaryTypes >
    {
        public:
            KGBEMMeshConverter();
            KGBEMMeshConverter( KSurfaceContainer& aContainer );
            virtual ~KGBEMMeshConverter();

        protected:
            void DispatchSurface( KGSurface* aSurface );
            void DispatchSpace( KGSpace* aSpace );

        private:
            void Add( KGMeshData* aData );

    };

    class KGBEMAxialMeshConverter :
        public KGDualHierarchy< KGBEMConverterNode, KBasisTypes, KBoundaryTypes >
    {
        public:
            KGBEMAxialMeshConverter();
            KGBEMAxialMeshConverter( KSurfaceContainer& aContainer );
            virtual ~KGBEMAxialMeshConverter();

        protected:
            void DispatchSurface( KGSurface* aSurface );
            void DispatchSpace( KGSpace* aSpace );

        private:
            void Add( KGAxialMeshData* aData );

    };

    class KGBEMDiscreteRotationalMeshConverter :
        public KGDualHierarchy< KGBEMConverterNode, KBasisTypes, KBoundaryTypes >
    {
        public:
            KGBEMDiscreteRotationalMeshConverter();
            KGBEMDiscreteRotationalMeshConverter( KSurfaceContainer& aContainer );
            virtual ~KGBEMDiscreteRotationalMeshConverter();

        protected:
            void DispatchSurface( KGSurface* aSurface );
            void DispatchSpace( KGSpace* aSpace );

        private:
            void Add( KGDiscreteRotationalMeshData* aData );

    };

}

#endif
