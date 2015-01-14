#ifndef Kassiopeia_KSRootSpace_h_
#define Kassiopeia_KSRootSpace_h_

#include "KSSpace.h"
#include "KSSurface.h"
#include "KSSide.h"

#include "KSList.h"

namespace Kassiopeia
{

    class KSRootSpace :
            public KSComponentTemplate< KSRootSpace, KSSpace >
    {
        public:
            KSRootSpace();
            KSRootSpace( const KSRootSpace& aCopy );
            KSRootSpace* Clone() const;
            virtual ~KSRootSpace();

        public:
            void Enter() const;
            void Exit() const;

            bool Outside( const KThreeVector& aPoint ) const;
            KThreeVector Point( const KThreeVector& aPoint ) const;
            KThreeVector Normal( const KThreeVector& aPoint ) const;

        public:
            void AddSpace( KSSpace* aSpace );
            void RemoveSpace( KSSpace* aSpace );

            void AddSurface( KSSurface* aSurface );
            void RemoveSurface( KSSurface* aSurface );

        protected:
            void InitializeComponent();
            void DeinitializeComponent();
    };

}

#endif
