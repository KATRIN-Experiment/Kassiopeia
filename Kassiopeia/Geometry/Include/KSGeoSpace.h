#ifndef Kassiopeia_KSGeoSpace_h_
#define Kassiopeia_KSGeoSpace_h_

#include "KSSpace.h"

#include "KGCore.hh"
using namespace KGeoBag;

namespace Kassiopeia
{

    class KSGeoSurface;
    class KSGeoSide;

    class KSGeoSpace :
        public KSComponentTemplate< KSGeoSpace, KSSpace >
    {

        public:
            KSGeoSpace();
            KSGeoSpace( const KSGeoSpace& aCopy );
            KSGeoSpace* Clone() const;
            virtual ~KSGeoSpace();

        public:
            void Enter() const;
            void Exit() const;

            bool Outside( const KThreeVector& aPoint ) const;
            KThreeVector Point( const KThreeVector& aPoint ) const;
            KThreeVector Normal( const KThreeVector& aPoint ) const;

        public:
            void AddContent( KGSpace* aSpace );
            void RemoveContent( KGSpace* aSpace );
            std::vector< KGSpace* > GetContent();

            void AddCommand( KSCommand* anCommand );
            void RemoveCommand( KSCommand* anCommand );

        protected:
            void InitializeComponent();
            void DeinitializeComponent();

        private:
            mutable std::vector< KGSpace* > fContents;
            mutable std::vector< KSCommand* > fCommands;
    };

}

#endif
