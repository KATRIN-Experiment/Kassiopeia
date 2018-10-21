#ifndef Kommon_KConditionProcessor_hh_
#define Kommon_KConditionProcessor_hh_

#include "KProcessor.hh"

#include <stack>
#include <vector>
#include <sstream>

namespace katrin
{

    class KConditionProcessor :
        public KProcessor
    {
        private:
            typedef std::vector< KToken* > TokenVector;
            typedef TokenVector::iterator TokenIt;
            typedef TokenVector::const_iterator TokenCIt;

        public:
            KConditionProcessor();
            virtual ~KConditionProcessor();

            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KElementDataToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );

        private:
            void Dispatch( KToken* aToken );

            typedef enum
            {
                eElementInactive, eActive, eElseActive, eElementComplete = -1
            } ElementState;
            ElementState fElementState;

            typedef enum
            {
                eAttributeInactive, eCondition, eAttributeComplete = -1
            } AttributeState;
            AttributeState fAttributeState;

            typedef enum
            {
                eIfCondition, eElseCondition
            } ProcessorState;
            ProcessorState fProcessorState;

            unsigned int fNest;
            bool fCondition;
            TokenVector fIfTokens;
            TokenVector fElseTokens;
            KProcessor* fNewParent;
            KProcessor* fOldParent;
    };

}

#endif
