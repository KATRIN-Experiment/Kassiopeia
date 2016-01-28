#ifndef Kassiopeia_KSComponentMathBuilder_h_
#define Kassiopeia_KSComponentMathBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentMath.h"
#include "KSToolbox.h"
#include "KSObjectsMessage.h"
#include "KSComponentGroup.h"

using namespace Kassiopeia;
namespace katrin
{

    class KSComponentMathData
    {
        public:
            string fName;
            string fGroupName;
            string fTerm;
            vector< string > fParents;
    };

    KSComponent* BuildOutputMath( vector< KSComponent* > aComponents, string aTerm )
    {
        if( aComponents.at( 0 )->Is< unsigned short >() == true )
        {
            vector< unsigned short* > tComponents;
            tComponents.push_back( aComponents.at( 0 )->As< unsigned short >() );
            for( size_t tIndex = 1; tIndex < aComponents.size(); tIndex++ )
            {
                if( aComponents.at( tIndex )->Is< unsigned short >() == true )
                {
                    tComponents.push_back( aComponents.at( tIndex )->As< unsigned short >() );
                }
                else
                {
                    objctmsg( eError ) << "KSComponentMath does only support same types for all parents" << eom;
                    return NULL;
                }
            }
            return new KSComponentMath< unsigned short >( aComponents, tComponents, aTerm );
        }

        if( aComponents.at( 0 )->Is< short >() == true )
        {
            vector< short* > tComponents;
            tComponents.push_back( aComponents.at( 0 )->As< short >() );
            for( size_t tIndex = 1; tIndex < aComponents.size(); tIndex++ )
            {
                if( aComponents.at( tIndex )->Is< short >() == true )
                {
                    tComponents.push_back( aComponents.at( tIndex )->As< short >() );
                }
                else
                {
                    objctmsg( eError ) << "KSComponentMath does only support same types for all parents" << eom;
                    return NULL;
                }
            }
            return new KSComponentMath< short >( aComponents, tComponents, aTerm );
        }

        if( aComponents.at( 0 )->Is< unsigned int >() == true )
        {
            vector< unsigned int* > tComponents;
            tComponents.push_back( aComponents.at( 0 )->As< unsigned int >() );
            for( size_t tIndex = 1; tIndex < aComponents.size(); tIndex++ )
            {
                if( aComponents.at( tIndex )->Is< unsigned int >() == true )
                {
                    tComponents.push_back( aComponents.at( tIndex )->As< unsigned int >() );
                }
                else
                {
                    objctmsg( eError ) << "KSComponentMath does only support same types for all parents" << eom;
                    return NULL;
                }
            }
            return new KSComponentMath< unsigned int >( aComponents, tComponents, aTerm );
        }

        if( aComponents.at( 0 )->Is< int >() == true )
        {
            vector< int* > tComponents;
            tComponents.push_back( aComponents.at( 0 )->As< int >() );
            for( size_t tIndex = 1; tIndex < aComponents.size(); tIndex++ )
            {
                if( aComponents.at( tIndex )->Is< int >() == true )
                {
                    tComponents.push_back( aComponents.at( tIndex )->As< int >() );
                }
                else
                {
                    objctmsg( eError ) << "KSComponentMath does only support same types for all parents" << eom;
                    return NULL;
                }
            }
            return new KSComponentMath< int >( aComponents, tComponents, aTerm );
        }

        if( aComponents.at( 0 )->Is< unsigned long >() == true )
        {
            vector< unsigned long* > tComponents;
            tComponents.push_back( aComponents.at( 0 )->As< unsigned long >() );
            for( size_t tIndex = 1; tIndex < aComponents.size(); tIndex++ )
            {
                if( aComponents.at( tIndex )->Is< unsigned long >() == true )
                {
                    tComponents.push_back( aComponents.at( tIndex )->As< unsigned long >() );
                }
                else
                {
                    objctmsg( eError ) << "KSComponentMath does only support same types for all parents" << eom;
                    return NULL;
                }
            }
            return new KSComponentMath< unsigned long >( aComponents, tComponents, aTerm );
        }

        if( aComponents.at( 0 )->Is< long >() == true )
        {
            vector< long* > tComponents;
            tComponents.push_back( aComponents.at( 0 )->As< long >() );
            for( size_t tIndex = 1; tIndex < aComponents.size(); tIndex++ )
            {
                if( aComponents.at( tIndex )->Is< long >() == true )
                {
                    tComponents.push_back( aComponents.at( tIndex )->As< long >() );
                }
                else
                {
                    objctmsg( eError ) << "KSComponentMath does only support same types for all parents" << eom;
                    return NULL;
                }
            }
            return new KSComponentMath< long >( aComponents, tComponents, aTerm );
        }

        if( aComponents.at( 0 )->Is< float >() == true )
        {
            vector< float* > tComponents;
            tComponents.push_back( aComponents.at( 0 )->As< float >() );
            for( size_t tIndex = 1; tIndex < aComponents.size(); tIndex++ )
            {
                if( aComponents.at( tIndex )->Is< float >() == true )
                {
                    tComponents.push_back( aComponents.at( tIndex )->As< float >() );
                }
                else
                {
                    objctmsg( eError ) << "KSComponentMath does only support same types for all parents" << eom;
                    return NULL;
                }
            }
            return new KSComponentMath< float >( aComponents, tComponents, aTerm );
        }

        if( aComponents.at( 0 )->Is< double >() == true )
        {
            vector< double* > tComponents;
            tComponents.push_back( aComponents.at( 0 )->As< double >() );
            for( size_t tIndex = 1; tIndex < aComponents.size(); tIndex++ )
            {
                if( aComponents.at( tIndex )->Is< double >() == true )
                {
                    tComponents.push_back( aComponents.at( tIndex )->As< double >() );
                }
                else
                {
                    objctmsg( eError ) << "KSComponentMath does only support same types for all parents" << eom;
                    return NULL;
                }
            }
            return new KSComponentMath< double >( aComponents, tComponents, aTerm );
        }

        objctmsg( eError ) << "KSComponentMathBuilder does only support int and double like types" << eom;
        return NULL;
    }

    typedef KComplexElement< KSComponentMathData > KSComponentMathBuilder;

    template< >
    inline bool KSComponentMathBuilder::Begin()
    {
        fObject = new KSComponentMathData;
        return true;
    }

    template< >
    inline bool KSComponentMathBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            string tName = aContainer->AsReference< string >();
            fObject->fName = tName;
            return true;
        }
        if( aContainer->GetName() == "group" )
        {
            string tGroupName = aContainer->AsReference< string >();
            fObject->fGroupName = tGroupName;
            return true;
        }
        if( aContainer->GetName() == "term" )
        {
            string tTerm = aContainer->AsReference< string >();
            fObject->fTerm = tTerm;
            return true;
        }
        if( aContainer->GetName() == "component" )
        {
            objctmsg( eWarning ) <<"deprecated warning in KSComponentMathBuilder: Please use the attribute <parent> instead <component>"<<eom;
            string tComponent = aContainer->AsReference< string >();
            fObject->fParents.push_back( tComponent );
            return true;
        }
        if( aContainer->GetName() == "parent" )
        {
            string tComponent = aContainer->AsReference< string >();
            fObject->fParents.push_back( tComponent );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSComponentMathBuilder::End()
    {
        vector< KSComponent* > tParentComponents;
        if( !fObject->fGroupName.empty() )
        {
            KSComponentGroup* tComponentGroup = KSToolbox::GetInstance()->GetObjectAs< KSComponentGroup >( fObject->fGroupName );
            for( size_t tNameIndex = 0; tNameIndex < fObject->fParents.size(); tNameIndex++ )
            {
                KSComponent* tOneComponent = NULL;
                for( unsigned int tGroupIndex = 0; tGroupIndex < tComponentGroup->ComponentCount(); tGroupIndex++ )
                {
                    KSComponent* tGroupComponent = tComponentGroup->ComponentAt( tGroupIndex );
                    if( tGroupComponent->GetName() == fObject->fParents.at( tNameIndex ) )
                    {
                        tOneComponent = tGroupComponent;
                        break;
                    }
                }
                if( tOneComponent == NULL )
                {
                    objctmsg( eError ) << "KSComponentMathBuilder can not find component < " << fObject->fParents.at( tNameIndex ) << " > in group < " << fObject->fGroupName << " >" << eom;
                }
                tParentComponents.push_back( tOneComponent );
            }
        }
        else
        {
            for( size_t tIndex = 0; tIndex < fObject->fParents.size(); tIndex++ )
            {
                KSComponent* tOneComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( fObject->fParents.at( tIndex ) );
                tParentComponents.push_back( tOneComponent );
            }
        }
        KSComponent* tComponent = BuildOutputMath( tParentComponents, fObject->fTerm );
        tComponent->SetName( fObject->fName );
        delete fObject;
        Set( tComponent );
        return true;
    }

}

#endif
