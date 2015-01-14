#ifndef Kassiopeia_KSTermMaxLength_h_
#define Kassiopeia_KSTermMaxLength_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMaxLength :
        public KSComponentTemplate< KSTermMaxLength, KSTerminator >
    {
        public:
            KSTermMaxLength();
            KSTermMaxLength( const KSTermMaxLength& aCopy );
            KSTermMaxLength* Clone() const;
            virtual ~KSTermMaxLength();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetLength( const double& aValue );

        private:
            double fLength;
    };

    inline void KSTermMaxLength::SetLength( const double& aValue )
    {
        fLength = aValue;
    }

}

#endif
