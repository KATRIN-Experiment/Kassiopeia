#ifndef Kassiopeia_KSSurface_h_
#define Kassiopeia_KSSurface_h_

#include "KSComponentTemplate.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

    class KSSpace;

    class KSSurface :
        public KSComponentTemplate< KSSurface >
    {
        public:
            friend class KSSpace;

        public:
            KSSurface();
            virtual ~KSSurface();

        public:
            virtual void On() const = 0;
            virtual void Off() const = 0;

            virtual KThreeVector Point( const KThreeVector& aPoint ) const = 0;
            virtual KThreeVector Normal( const KThreeVector& aPoint ) const = 0;

            const KSSpace* GetParent() const;
            KSSpace* GetParent();
            void SetParent( KSSpace* aSpace );

        protected:
            KSSpace* fParent;
    };

}

#endif
