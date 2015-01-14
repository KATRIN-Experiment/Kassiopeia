#ifndef KGPLANARPOLYLINE_HH_
#define KGPLANARPOLYLINE_HH_

#include "KGPlanarOpenPath.hh"
#include "KGPlanarLineSegment.hh"
#include "KGPlanarArcSegment.hh"

namespace KGeoBag
{

    class KGPlanarPolyLine :
        public KGPlanarOpenPath
    {
        public:
            typedef deque< const KGPlanarOpenPath* > Set;
            typedef Set::iterator It;
            typedef Set::const_iterator CIt;

        public:
            KGPlanarPolyLine();
            KGPlanarPolyLine( const KGPlanarPolyLine& aCopy );
            virtual ~KGPlanarPolyLine();

            KGPlanarPolyLine* Clone() const;
            void CopyFrom( const KGPlanarPolyLine& aCopy );

        public:
            void StartPoint( const KTwoVector& aPoint );
            void NextLine( const KTwoVector& aVertex, const unsigned int aCount = 1, const double aPower = 1. );
            void NextArc( const KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount = 1 );
            void PreviousLine( const KTwoVector& aVertex, const unsigned int aCount = 1, const double aPower = 1. );
            void PreviousArc( const KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount = 1 );

            const Set& Elements() const;

            const double& Length() const;
            const KTwoVector& Centroid() const;
            const KTwoVector& Start() const;
            const KTwoVector& End() const;

        public:
            KTwoVector At( const double& aLength ) const;
            KTwoVector Point( const KTwoVector& aQuery ) const;
            KTwoVector Normal( const KTwoVector& aQuery ) const;
            bool Above( const KTwoVector& aQuery ) const;

        private:
            Set fElements;

            mutable double fLength;
            mutable KTwoVector fCentroid;
            mutable KTwoVector fStart;
            mutable KTwoVector fEnd;

            void Initialize() const;
            mutable bool fInitialized;

        public:
            class StartPointArguments
            {
                public:
                    StartPointArguments() :
                            fPoint( 0., 0. )
                    {
                    }
                    ~StartPointArguments()
                    {
                    }

                    KTwoVector fPoint;
            };

            class LineArguments
            {
                public:
                    LineArguments() :
                            fVertex( 0., 0. ),
                            fMeshCount( 1 ),
                            fMeshPower( 1. )
                    {
                    }
                    ~LineArguments()
                    {
                    }

                    KTwoVector fVertex;
                    unsigned int fMeshCount;
                    double fMeshPower;
            };

            class ArcArguments
            {
                public:
                    ArcArguments() :
                            fVertex( 0., 0. ),
                            fRadius( 0. ),
                            fRight( true ),
                            fShort( true ),
                            fMeshCount( 64 )
                    {
                    }
                    ~ArcArguments()
                    {
                    }

                    KTwoVector fVertex;
                    double fRadius;
                    bool fRight;
                    bool fShort;
                    unsigned int fMeshCount;
            };
    };

}

#endif
