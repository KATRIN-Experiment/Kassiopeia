#ifndef KSROOTQUANTITY_H_
#define KSROOTQUANTITY_H_

#include "KSWriter.h"

#include "KSList.h"

namespace Kassiopeia
{

    class KSRootWriter :
        public KSComponentTemplate< KSRootWriter, KSWriter >
    {
        public:
            KSRootWriter();
            KSRootWriter( const KSRootWriter& aCopy );
            KSRootWriter* Clone() const;
            virtual ~KSRootWriter();

            //******
            //writer
            //******

        public:
            void ExecuteRun();
            void ExecuteEvent();
            void ExecuteTrack();
            void ExecuteStep();

            //***********
            //composition
            //***********

        public:
            void AddWriter( KSWriter* aWriter );
            void RemoveWriter( KSWriter* aWriter );

        private:
            KSList< KSWriter > fWriters;
    };
}

#endif
