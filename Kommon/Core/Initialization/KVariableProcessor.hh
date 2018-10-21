#ifndef Kommon_KVariableProcessor_hh_
#define Kommon_KVariableProcessor_hh_

#include "KProcessor.hh"

#include <stack>
#include <map>

namespace katrin
{

    class KVariableProcessor :
        public KProcessor
    {
        private:
            typedef std::map< std::string, std::string > VariableMap;
            typedef VariableMap::value_type VariableEntry;
            typedef VariableMap::iterator VariableIt;
            typedef VariableMap::const_iterator VariableCIt;

            typedef std::map< std::string, std::uint32_t > ReferenceCountMap;
            typedef ReferenceCountMap::value_type ReferenceCountEntry;

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
                /*  0 */ eElementInactive,
                /*  1 */ eActiveLocalDefine,eActiveGlobalDefine, eActiveExternalDefine,
                /*  4 */ eActiveLocalRedefine, eActiveGlobalRedefine, eActiveExternalRedefine,
                /*  7 */ eActiveLocalUndefine, eActiveGlobalUndefine, eActiveExternalUndefine,
                /* 10 */ eActiveLocalAppend, eActiveGlobalAppend, eActiveExternalAppend,
                /* -1 */ eElementComplete = -1
            } ElementState;
            typedef enum
            {
                /*  0 */ eAttributeInactive,
                /*  1 */ eActiveName, eActiveValue,
                /* -1 */ eAttributeComplete = -1
            } AttributeState;

            ElementState fElementState;
            AttributeState fAttributeState;

            std::string fName;
            std::string fValue;
            ReferenceCountMap fRefCountMap; // need only one instance since variable names are unique
            VariableMap* fExternalMap;
            VariableMap* fGlobalMap;
            VariableMap* fLocalMap;
            std::stack< VariableMap* > fLocalMapStack;

            static const std::string fStartBracket;
            static const std::string fEndBracket;
            static const std::string fNameValueSeparator;
            static const std::string fAppendValueSeparator;
    };

}

#endif
