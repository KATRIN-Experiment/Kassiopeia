#ifndef Kommon_KVariableProcessor_hh_
#define Kommon_KVariableProcessor_hh_

#include "KProcessor.hh"

#include <stack>
using std::stack;

#include <map>
using std::map;

namespace katrin
{

    class KVariableProcessor :
        public KProcessor
    {
        private:
            typedef map< string, string > VariableMap;
            typedef VariableMap::value_type VariableEntry;
            typedef VariableMap::iterator VariableIt;
            typedef VariableMap::const_iterator VariableCIt;

        public:
            KVariableProcessor();
            KVariableProcessor( const VariableMap& anExternalMap );
            virtual ~KVariableProcessor();

            virtual void ProcessToken( KBeginFileToken* aToken );
            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KElementDataToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );
            virtual void ProcessToken( KEndFileToken* aToken );

        private:
            void Evaluate( KToken* aToken );

            typedef enum
            {
                eElementInactive, eActiveLocalDefine, eActiveGlobalDefine, eActiveExternalDefine, eActiveLocalUndefine, eActiveGlobalUndefine, eActiveExternalUndefine, eElementComplete
            } ElementState;
            typedef enum
            {
                eAttributeInactive, eActiveName, eActiveValue, eAttributeComplete
            } AttributeState;

            ElementState fElementState;
            AttributeState fAttributeState;

            string fName;
            string fValue;
            VariableMap* fExternalMap;
            VariableMap* fGlobalMap;
            VariableMap* fLocalMap;
            stack< VariableMap* > fLocalMapStack;

            static const string fStartBracket;
            static const string fEndBracket;
            static const string fNameValueSeparator;
    };

}

#endif
