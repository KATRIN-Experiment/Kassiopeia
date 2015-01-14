#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KElementProcessor.hh"

#include "KAttribute.hh"
#include "KComplexElement.hh"

#include <iostream>
using std::cout;
using std::endl;

namespace katrin
{

    class TestChild
    {
        public:
            TestChild() :
                fName( "(anonymous)" ),
                fValue( 0 )
            {
            }
            virtual ~TestChild()
            {
                cout << "a test child is destroyed" << endl;
            }

            void SetName( const string& aName )
            {
                fName = aName;
                return;
            }

            void SetValue( const int& aValue )
            {
                fValue = aValue;
                return;
            }

            void Print() const
            {
                cout << "  *" << fName << ", " << fValue << endl;
                return;
            }

        private:
            string fName;
            int fValue;
    };

    typedef KComplexElement< TestChild > KiTestChildElement;

    template<>
    bool KiTestChildElement::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &TestChild::SetName );
            return true;
        }
        if( anAttribute->GetName() == "value" )
        {
            anAttribute->CopyTo( fObject, &TestChild::SetValue );
            return true;
        }
        return false;
    }

    template<>
    bool KiTestChildElement::End()
    {
        fObject->Print();
        return true;
    }

    static const int sTestChildElementStructure =
        KiTestChildElement::Attribute< string >( "name" ) +
        KiTestChildElement::Attribute< int >( "value" );

    class TestParent :
        public KContainer
    {
        public:
            TestParent()
            {
            }
            virtual ~TestParent()
            {
                for( vector< const TestChild* >::const_iterator tIter = fChildren.begin(); tIter != fChildren.end(); tIter++ )
                {
                    delete *tIter;
                }
                cout << "a test parent is destroyed" << endl;
            }

            void SetName( const string& aName )
            {
                fName = aName;
                return;
            }
            void AddChild( const TestChild* aChild )
            {
                fChildren.push_back( aChild );
            }

            void Print() const
            {
                cout << "*" << fName << endl;
                for( vector< const TestChild* >::const_iterator tIter = fChildren.begin(); tIter != fChildren.end(); tIter++ )
                {
                    (*tIter)->Print();
                }
                return;
            }

        private:
            string fName;
            vector< const TestChild* > fChildren;
    };

    typedef KComplexElement< TestParent > KiTestParentElement;

    template<>
    bool KiTestParentElement::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &TestParent::SetName );
            return true;
        }
        return false;
    }

    template<>
    bool KiTestParentElement::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "child" )
        {
            anElement->ReleaseTo( fObject, &TestParent::AddChild );
            return true;
        }
        return false;
    }

    template<>
    bool KiTestParentElement::End()
    {
        fObject->Print();
        return true;
    }

    static const int sTestParentElementStructure =
        KiTestParentElement::Attribute< string >( "name" ) +
        KiTestParentElement::ComplexElement< TestChild >( "child" );

    static const int sTestParentElement =
        KElementProcessor::ComplexElement< TestParent >( "parent" );

}

using namespace katrin;
using namespace katrin;

int main()
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance()->SetLogVerbosity( eDebug );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLElements.xml" );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tFormulaProcessor.InsertAfter( &tVariableProcessor );
    tIncludeProcessor.InsertAfter( &tFormulaProcessor );
    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tElementProcessor.InsertAfter( &tLoopProcessor );

    tTokenizer.ProcessFile( tFile );

    return 0;
}
