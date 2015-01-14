#ifndef Kommon_KConditionProcessor_hh_
#define Kommon_KConditionProcessor_hh_

#include "KProcessor.hh"

#include <stack>
using std::stack;

#include <vector>
using std::vector;

#include <sstream>
using std::stringstream;

#include <cstdlib>

namespace katrin
{

    class KConditionProcessor :
        public KProcessor
    {
        private:
            typedef vector< KToken* > TokenVector;
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
                eElementInactive, eActive, eElementComplete
            } ElementState;
            ElementState fElementState;

            typedef enum
            {
                eAttributeInactive, eCondition, eAttributeComplete
            } AttributeState;
            AttributeState fAttributeState;

            unsigned int fNest;
            bool fCondition;
            TokenVector fIfTokens;
            KProcessor* fNewParent;
            KProcessor* fOldParent;
    };

}

#endif
