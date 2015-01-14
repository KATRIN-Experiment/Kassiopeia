#ifndef Kommon_KTagProcessor_hh_
#define Kommon_KTagProcessor_hh_

#include "KProcessor.hh"

#include <stack>
using std::stack;

#include <vector>
using std::vector;

namespace katrin
{

    class KTagProcessor :
        public KProcessor
    {
        public:
            KTagProcessor();
            virtual ~KTagProcessor();

            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );

        private:
            typedef enum
            {
                eInactive, eActive
            } State;
            State fState;

            vector< string > fTags;
            vector< string >::iterator fTagIt;
            stack< vector< string > > fTagStack;
    };

}

#endif
