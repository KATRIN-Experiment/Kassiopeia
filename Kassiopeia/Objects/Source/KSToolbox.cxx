#include "KSToolbox.h"

namespace Kassiopeia
{

    KSToolbox::KSToolbox() :
            fNameMap(),
            fTagMap()
    {
        fTagMap.insert( TagMapEntry( string( "" ), new ObjectSet() ) );
    }
    KSToolbox::~KSToolbox()
    {
        NameMapIt tNameIt;
        for( tNameIt = fNameMap.begin(); tNameIt != fNameMap.end(); tNameIt++ )
        {
            objctmsg_debug( "deleting object with name <" << tNameIt->first << ">" << eom )
            delete tNameIt->second;
        }

        TagMapIt tTagIt;
        for( tTagIt = fTagMap.begin(); tTagIt != fTagMap.end(); tTagIt++ )
        {
            objctmsg_debug( "deleting object set with tag <" << tTagIt->first << ">" << eom )
            delete tTagIt->second;
        }
    }

    void KSToolbox::AddObject( KSObject* aObject )
    {
        NameMapIt tIter = fNameMap.find( aObject->GetName() );
        if( tIter == fNameMap.end() )
        {
            objctmsg_debug( "adding object with name <" << aObject->GetName() << ">" << eom )
            fNameMap.insert( NameMapEntry( aObject->GetName(), aObject ) );

            const KTagSet& aTagSet = aObject->GetTags();
            KTagSetIt tTagSetIt;

            TagMapIt tTagMapIt;
            ObjectSet* tObjectSet;
            for( tTagSetIt = aTagSet.begin(); tTagSetIt != aTagSet.end(); tTagSetIt++ )
            {
                tTagMapIt = fTagMap.find( *tTagSetIt );
                if( tTagMapIt == fTagMap.end() )
                {
                    objctmsg_debug( "adding object set for tag <" << *tTagSetIt << ">" << eom )
                    tObjectSet = new ObjectSet();
                    fTagMap.insert( TagMapEntry( *tTagSetIt, tObjectSet ) );
                }
                else
                {
                    tObjectSet = tTagMapIt->second;
                }
                tObjectSet->insert( aObject );
            }
            return;
        }
        objctmsg( eError ) << "multiple instances of object with name <" << aObject->GetName() << ">" << eom;
        return;
    }
    void KSToolbox::RemoveObject( KSObject* aObject )
    {
        NameMapIt tIter = fNameMap.find( aObject->GetName() );
        if( tIter != fNameMap.end() )
        {
            objctmsg_debug( "removing object with name <" << aObject->GetName() << ">" << eom )
            fNameMap.erase( tIter );

            const KTagSet& aTagSet = aObject->GetTags();
            KTagSetIt tTagSetIt;

            TagMapIt tTagMapIt;
            ObjectSet* tObjectSet;
            for( tTagSetIt = aTagSet.begin(); tTagSetIt != aTagSet.end(); tTagSetIt++ )
            {
                tTagMapIt = fTagMap.find( *tTagSetIt );
                if( tTagMapIt != fTagMap.end() )
                {
                    tObjectSet = tTagMapIt->second;
                    tObjectSet->erase( aObject );
                    if( tObjectSet->empty() == true )
                    {
                        objctmsg_debug( "removing object set for tag <" << *tTagSetIt << ">" << eom )
                        delete tObjectSet;
                        fTagMap.erase( tTagMapIt );
                    }
                }
            }
            return;
        }
        objctmsg( eError ) << "no instances of object with name <" << aObject->GetName() << ">" << eom;
        return;
    }

    KSObject* KSToolbox::GetObject( const string& aName )
    {
        NameMapIt tIter = fNameMap.find( aName );
        if( tIter != fNameMap.end() )
        {
            return tIter->second;
        }
        objctmsg( eError ) << "no instances of object with name <" << aName << ">" << eom;
        return NULL;
    }

    bool KSToolbox::HasObject( const string& aName )
    {
        NameMapIt tIter = fNameMap.find( aName );
        return ( tIter != fNameMap.end() );
    }

    KSToolbox::ObjectSet KSToolbox::GetObjects( const string& aTag )
    {
        TagMapIt tIter = fTagMap.find( aTag );
        if( tIter != fTagMap.end() )
        {
            return *( tIter->second );
        }
        objctmsg( eError ) << "no instances of object with tag <" << aTag << ">" << eom;
        return ObjectSet();
    }

}
