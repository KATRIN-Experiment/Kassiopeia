#ifndef Kassiopeia_KSGenTimeComposite_h_
#define Kassiopeia_KSGenTimeComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

    class KSGenTimeComposite :
        public KSComponentTemplate< KSGenTimeComposite, KSGenCreator >
    {
        public:
            KSGenTimeComposite();
            KSGenTimeComposite( const KSGenTimeComposite& aCopy );
            KSGenTimeComposite* Clone() const;
            virtual ~KSGenTimeComposite();

        public:
            virtual void Dice( KSParticleQueue* aPrimaries );

        public:
            void SetTimeValue( KSGenValue* anTimeValue );
            void ClearTimeValue( KSGenValue* anTimeValue );

        private:
            KSGenValue* fTimeValue;

        protected:
            void InitializeComponent();
            void DeinitializeComponent();

    };
}

#endif
