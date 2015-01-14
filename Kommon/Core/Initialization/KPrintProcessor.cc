#include "KPrintProcessor.hh"
#include "KInitializationMessage.hh"

#include <iostream>
using std::cout;
using std::endl;

#include <cstdlib>

namespace katrin
{

    KPrintProcessor::KPrintProcessor() :
            KProcessor(),
            fElementState( eElementInactive ),
            fAttributeState( eAttributeInactive ),
            fName( "" ),
            fValue( "" )
    {
    }

    KPrintProcessor::~KPrintProcessor()
    {
    }

    void KPrintProcessor::ProcessToken( KBeginElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            if( aToken->GetValue() == string( "print" ) )
            {
                fElementState = eElementActive;
                return;
            }

            KProcessor::ProcessToken( aToken );
            return;
        }
        return;
    }

    void KPrintProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementActive )
        {
            if( aToken->GetValue() == "name" )
            {
                fAttributeState = eActiveName;
                return;
            }
            if( aToken->GetValue() == "value" )
            {
                fAttributeState = eActiveValue;
                return;
            }

            initmsg( eError ) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;

            return;
        }
        return;
    }

    void KPrintProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementActive )
        {
            if( fAttributeState == eActiveName )
            {
                fName = aToken->GetValue< string >();
                fAttributeState = eAttributeComplete;
                return;
            }
            if( fAttributeState == eActiveValue )
            {
                fValue = aToken->GetValue< string >();
                fAttributeState = eAttributeComplete;
                return;
            }
        }
        return;
    }

    void KPrintProcessor::ProcessToken( KEndAttributeToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementActive )
        {
            if( fAttributeState == eAttributeComplete )
            {
                fAttributeState = eAttributeInactive;
                return;
            }
        }
        return;
    }

    void KPrintProcessor::ProcessToken( KMidElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementActive )
        {
            fElementState = eElementComplete;
            return;
        }
        return;
    }

    void KPrintProcessor::ProcessToken( KElementDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            initmsg( eError ) << "got unknown element data <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            return;
        }

        return;
    }

    void KPrintProcessor::ProcessToken( KEndElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            initmsg( eNormal ) << "value of <" << fName << "> is <" << fValue << ">" << eom;
            fName.clear();
            fValue.clear();
            fElementState = eElementInactive;
            return;
        }

        return;
    }

}
