#ifndef Kommon_KPrintProcessor_hh_
#define Kommon_KPrintProcessor_hh_

#include "KProcessor.hh"

#include <stack>
#include <map>

namespace katrin
{

    class KPrintProcessor :
        public KProcessor
    {

        public:
    		KPrintProcessor();
            virtual ~KPrintProcessor();

            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KElementDataToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );

        private:
            typedef enum
            {
                eElementInactive, eElementActive, eElementComplete
            } ElementState;
            typedef enum
            {
                eAttributeInactive, eActiveName, eActiveValue, eAttributeComplete
            } AttributeState;

            ElementState fElementState;
            AttributeState fAttributeState;
            KMessageSeverity fMessageType;

            std::string fName;
            std::string fValue;
    };

}

#endif
