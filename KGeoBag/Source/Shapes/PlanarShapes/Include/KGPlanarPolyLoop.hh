#ifndef KGPLANARPOLYLOOP_HH_
#define KGPLANARPOLYLOOP_HH_

#include "KGPlanarClosedPath.hh"
#include "KGPlanarLineSegment.hh"
#include "KGPlanarArcSegment.hh"

#include <list>
using std::list;

namespace KGeoBag
{

    class KGPlanarPolyLoop :
        public KGPlanarClosedPath
    {
        public:
            typedef deque< const KGPlanarOpenPath* > Set;
            typedef Set::iterator It;
            typedef Set::const_iterator CIt;

        public:
            KGPlanarPolyLoop();
            KGPlanarPolyLoop( const KGPlanarPolyLoop& aCopy );
            virtual ~KGPlanarPolyLoop();

            KGPlanarPolyLoop* Clone() const;
            void CopyFrom( const KGPlanarPolyLoop& aCopy );

        public:
            void StartPoint( const KTwoVector& aPoint );
            void NextLine( const KTwoVector& aVertex, const unsigned int aCount = 1, const double aPower = 1. );
            void NextArc( const KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount = 1 );
            void PreviousLine( const KTwoVector& aVertex, const unsigned int aCount = 1, const double aPower = 1. );
            void PreviousArc( const KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount = 1 );
            void LastLine( const unsigned int aCount = 1, const double aPower = 1. );
            void LastArc( const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount = 1 );

            const Set& Elements() const;

            const double& Length() const;
            const KTwoVector& Centroid() const;
            const KTwoVector& Anchor() const;

        public:
            KTwoVector At( const double& aLength ) const;
            KTwoVector Point( const KTwoVector& aQuery ) const;
            KTwoVector Normal( const KTwoVector& aQuery ) const;
            bool Above( const KTwoVector& aQuery ) const;

        private:
            Set fElements;

            mutable double fLength;
            mutable KTwoVector fCentroid;
            mutable KTwoVector fAnchor;

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

            class LastLineArguments
            {
                public:
                    LastLineArguments() :
                            fMeshCount( 1 ),
                            fMeshPower( 1. )
                    {
                    }
                    ~LastLineArguments()
                    {
                    }

                    unsigned int fMeshCount;
                    double fMeshPower;
            };

            class LastArcArguments
            {
                public:
                    LastArcArguments() :
                            fRadius( 0. ),
                            fRight( true ),
                            fShort( true ),
                            fMeshCount( 64 )
                    {
                    }
                    ~LastArcArguments()
                    {
                    }

                    double fRadius;
                    bool fRight;
                    bool fShort;
                    unsigned int fMeshCount;
            };
    };

}

#endif
