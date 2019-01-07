#include "KElementBase.hh"
#include "KAttribute.hh"
#include "KInitializationMessage.hh"

namespace katrin
{
    KElementBase::KElementBase() :
            fParentElement( NULL ),
            fAttributes( NULL ),
            fChildAttribute( NULL ),
            fAttributeDepth( 0 ),
            fElements( NULL ),
            fChildElement( NULL ),
            fElementDepth( 0 )
    {
    }
    KElementBase::~KElementBase()
    {
    }

    void KElementBase::ProcessToken( KBeginElementToken* aToken )
    {
        if( (fElementDepth == 0) && (fAttributeDepth == 0) )
        {
            //look up constructor method in the map, complain and exit if not found
            KElementCIt It = fElements->find( aToken->GetValue() );
            if( It == fElements->end() )
            {
                initmsg( eError ) << "nothing registered for element <" << aToken->GetValue() << "> in element <" << GetName() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
                return;
            }

            //construct and label element
            KElementBase* tChildElement = (It->second)( this );
            tChildElement->SetName( aToken->GetValue() );

            //begin element
            if( tChildElement->Begin() == false )
            {
                initmsg( eError ) << "could not begin element <" << aToken->GetValue() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
                return;
            }

            fChild = tChildElement;
            fChildElement = tChildElement;
            fElementDepth = 1;
            return;
        }
        else
        {
            fElementDepth++;
            KProcessor::ProcessToken( aToken );
            return;
        }
    }
    void KElementBase::ProcessToken( KBeginAttributeToken* aToken )
    {
        if( (fElementDepth == 0) && (fAttributeDepth == 0) )
        {
            //look up constructor method in the map, complain and exit if not found
            KAttributeCIt It = fAttributes->find( aToken->GetValue() );
            if( It == fAttributes->end() )
            {
                initmsg( eError ) << "nothing registered for attribute <" << aToken->GetValue() << "> in element <" << GetName() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
                return;
            }

            //construct attribute
            KAttributeBase* tChildAttribute = (It->second)( this );

            //label attribute
            tChildAttribute->SetName( aToken->GetValue() );

            fChild = tChildAttribute;
            fChildAttribute = tChildAttribute;
            fAttributeDepth = 1;
            return;
        }
        else
        {
            KProcessor::ProcessToken( aToken );
            return;
        }
    }
    void KElementBase::ProcessToken( KEndAttributeToken* aToken )
    {
        if( (fAttributeDepth == 1) && (fElementDepth == 0) )
        {
            //add attribute to this element
            if( AddAttribute( fChildAttribute ) == false )
            {
                initmsg( eError ) << "element <" << GetName() << "> could not process attribute <" << fChildAttribute->GetName() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
            }

            //delete attribute
            delete fChildAttribute;

            //reset child information
            fChild = NULL;
            fChildAttribute = NULL;
            fAttributeDepth = 0;
            return;
        }
        else
        {
            KProcessor::ProcessToken( aToken );
            return;
        }
    }
    void KElementBase::ProcessToken( KMidElementToken* aToken )
    {
        if( (fAttributeDepth == 0) && (fElementDepth == 1) )
        {
            //start body of child element
            if( fChildElement->Body() == false )
            {
                initmsg( eError ) << "could not begin body of element <" << aToken->GetValue() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
                return;
            }

            //nothing else to do
            return;
        }
        else
        {
            KProcessor::ProcessToken( aToken );
            return;
        }
    }
    void KElementBase::ProcessToken( KElementDataToken* aToken )
    {
        if( fElementDepth == 0 )
        {
            //add value to this element
            if( SetValue( aToken ) == false )
            {
                initmsg( eError ) << "element <" << GetName() << "> could not process value <" << aToken->GetValue() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
                return;
            }

            return;
        }
        else
        {
            KProcessor::ProcessToken( aToken );
        }
    }
    void KElementBase::ProcessToken( KEndElementToken* aToken )
    {
        if( (fAttributeDepth == 0) && (fElementDepth == 1) )
        {
            //end child element
            if( fChildElement->End() == false )
            {
                initmsg( eError ) << "could not end child element <" << aToken->GetValue() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
                return;
            }

            //add child element to this element
            if( AddElement( fChildElement ) == false )
            {
                initmsg( eError ) << "element <" << GetName() << "> could not process element <" << aToken->GetValue() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
                return;
            }

            //delete child element
            delete fChildElement;

            fChild = NULL;
            fChildElement = NULL;
            fElementDepth = 0;
            return;
        }
        else
        {
            fElementDepth--;
            KProcessor::ProcessToken( aToken );
            return;
        }
    }
    void KElementBase::ProcessToken( KErrorToken* aToken )
    {
        if( fElementDepth == 0 && fAttributeDepth == 0 )
        {
            initmsg( eError ) << "element <" << GetName() << "> encountered an error <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << "> at column <" << aToken->GetColumn() << ">" << eom;
            return;
        }
        else
        {
            KProcessor::ProcessToken( aToken );
            return;
        }
    }

}

