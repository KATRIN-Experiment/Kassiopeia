#ifndef KGCORE_HH_
#error "do not include KGInterface.hh directly; include KGCore.hh instead."
#else

#include "KGCoreMessage.hh"

namespace KGeoBag
{

    class KGInterface
    {
        public:
            static KGInterface* GetInstance();
            static KGInterface* DeleteInstance();

        private:
            KGInterface();
            virtual ~KGInterface();

            static KGInterface* sInstance;

        public:
            static const char sSeparator;
            static const char sNest;
            static const char sTag;
            static const char sRecurse;
            static const char sWildcard;

            //*******
            //install
            //*******

        public:
            void InstallSpace( KGSpace* aSpace );
            void InstallSurface( KGSurface* aSurface );

            //********
            //retrieve
            //********

        public:
            vector< KGSurface* > RetrieveSurfaces();
            vector< KGSurface* > RetrieveSurfaces( string aPath );
            KGSurface* RetrieveSurface( string aPath );

            vector< KGSpace* > RetrieveSpaces();
            vector< KGSpace* > RetrieveSpaces( string aPath );
            KGSpace* RetrieveSpace( string aPath );

        private:
            void RetrieveSurfacesBySpecifier( vector< KGSurface* >& anAccumulator, KGSpace* aNode, string aSpecifier );
            void RetrieveSpacesBySpecifier( vector< KGSpace* >& anAccumulator, KGSpace* aNode, string aSpecifier );

            void RetrieveSurfacesByPath( vector< KGSurface* >& anAccumulator, KGSpace* aNode, string aPath );
            void RetrieveSpacesByPath( vector< KGSpace* >& anAccumulator, KGSpace* aNode, string aPath );

            void RetrieveSurfacesByName( vector< KGSurface* >& anAccumulator, KGSpace* aNode, string aName );
            void RetrieveSpacesByName( vector< KGSpace* >& anAccumulator, KGSpace* aNode, string aName );

            void RetrieveSurfacesByTag( vector< KGSurface* >& anAccumulator, KGSpace* aNode, string aTag, int aDepth );
            void RetrieveSpacesByTag( vector< KGSpace* >& anAccumulator, KGSpace* aNode, string aTag, int aDepth );

            void RetrieveSurfacesByWildcard( vector< KGSurface* >& anAccumulator, KGSpace* aNode, int aDepth );
            void RetrieveSpacesByWildcard( vector< KGSpace* >& anAccumulator, KGSpace* aNode, int aDepth );

            //*****
            //smell
            //*****

        public:
            KGSpace* Root() const;

        private:
            KGSpace* fRoot;
    };

}

#endif
