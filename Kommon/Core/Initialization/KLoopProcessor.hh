#ifndef Kommon_KLoopProcessor_hh_
#define Kommon_KLoopProcessor_hh_

#include "KProcessor.hh"

#include <stack>
using std::stack;

#include <vector>
using std::vector;

#include <sstream>
using std::stringstream;

namespace katrin
{

    class KLoopProcessor :
        public KProcessor
    {
        private:
            typedef vector< KToken* > TokenVector;
            typedef TokenVector::iterator TokenIt;
            typedef TokenVector::const_iterator TokenCIt;

        public:
            KLoopProcessor();
            virtual ~KLoopProcessor();

            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KElementDataToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );

        private:
            void Reset();
            void Evaluate( KToken* aToken, const string& aName, const string& aValue );
            void Dispatch( KToken* aToken );

            typedef enum
            {
                eElementInactive, eActive, eElementComplete
            } ElementState;
            ElementState fElementState;

            typedef enum
            {
                eAttributeInactive, eVariable, eStart, eEnd, eStep, eAttributeComplete
            } AttributeState;
            AttributeState fAttributeState;

            unsigned int fNest;
            string fVariable;
            int fStartValue;
            int fEndValue;
            int fStepValue;
            TokenVector fTokens;
            KProcessor* fNewParent;
            KProcessor* fOldParent;

            static const string fStartBracket;
            static const string fEndBracket;
    };

}

#endif
