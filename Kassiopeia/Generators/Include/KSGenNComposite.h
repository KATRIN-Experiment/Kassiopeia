#ifndef Kassiopeia_KSGenNComposite_h_
#define Kassiopeia_KSGenNComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

    class KSGenNComposite :
        public KSComponentTemplate< KSGenNComposite, KSGenCreator >
    {
        public:
            KSGenNComposite();
            KSGenNComposite( const KSGenNComposite& aCopy );
            KSGenNComposite* Clone() const;
            virtual ~KSGenNComposite();

        public:
            virtual void Dice( KSParticleQueue* aPrimaries );

        public:
            void SetNValue( KSGenValue* anNValue );
            void ClearNValue( KSGenValue* anNValue );

        private:
            KSGenValue* fNValue;

        protected:
            void InitializeComponent();
            void DeinitializeComponent();

    };
}

#endif
