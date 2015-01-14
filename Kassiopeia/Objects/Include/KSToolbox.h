#ifndef Kassiopeia_KSToolbox_h_
#define Kassiopeia_KSToolbox_h_

#include "KSObject.h"
#include "KSObjectsMessage.h"

#include "KSingleton.h"
using katrin::KSingleton;
using katrin::KTag;
using katrin::KTagSet;
using katrin::KTagSetIt;
using katrin::KTagSetCIt;
using katrin::KTagged;
using katrin::KNamed;

#include <set>
using std::set;

#include <map>
using std::map;

#include <vector>
using std::vector;

namespace Kassiopeia
{

    class KSToolbox :
        public KSingleton< KSToolbox >
    {
        public:
            friend class KSingleton< KSToolbox > ;

            typedef KSObject Object;

            typedef map< string, Object* > NameMap;
            typedef NameMap::iterator NameMapIt;
            typedef NameMap::const_iterator NameMapCIt;
            typedef NameMap::value_type NameMapEntry;

            typedef set< Object* > ObjectSet;
            typedef ObjectSet::iterator ObjectIt;
            typedef ObjectSet::const_iterator ObjectCIt;

            typedef map< string, ObjectSet* > TagMap;
            typedef TagMap::iterator TagMapIt;
            typedef TagMap::const_iterator TagMapCIt;
            typedef TagMap::value_type TagMapEntry;

        public:
            KSToolbox();
            virtual ~KSToolbox();

            void AddObject( Object* aObject );
            void RemoveObject( Object* aObject );

            template< class XType >
            XType* GetObjectAs( const string& aName );
            Object* GetObject( const string& aName );

            template< class XType >
            bool HasObjectAs( const string& aName );
            bool HasObject( const string& aName );

            template< class XType >
			vector< XType* > GetObjectsAs();

            template< class XType >
            vector< XType* > GetObjectsAs( const string& aTag );
            ObjectSet GetObjects( const string& aTag );

        protected:
            NameMap fNameMap;
            TagMap fTagMap;
    };

    template< class XType >
    inline XType* KSToolbox::GetObjectAs( const string& aName )
    {
        if( XType* tObject = dynamic_cast< XType* >( GetObject( aName ) ) )
        {
            return tObject;
        }
        objctmsg( eError ) << "cannot cast object <" << aName << "> to requested type" << eom;
        return NULL;
    }

    template< class XType >
    inline bool KSToolbox::HasObjectAs( const string& aName )
    {
        return ( HasObject(aName) && dynamic_cast< XType* >( GetObject( aName ) ) != 0 );
    }

    template< class XType >
    inline vector< XType* > KSToolbox::GetObjectsAs( )
    {
    	vector< XType* > tCastObjectSet;
        for( NameMap::const_iterator tIt = fNameMap.begin(); tIt != fNameMap.end(); ++tIt )
        {
            Object* tObject = tIt->second;
            if( XType* tCastObject = dynamic_cast< XType* >( tObject ) )
            {
                tCastObjectSet.push_back( tCastObject );
            }
        }
        return tCastObjectSet;
    }

    template< class XType >
    inline vector< XType* > KSToolbox::GetObjectsAs( const string& aTag )
    {
        TagMapIt tTagMapIter = fTagMap.find( aTag );
        if( tTagMapIter == fTagMap.end() )
        {
            objctmsg( eError ) << "no instances of object with tag <" << aTag << ">" << eom;
            return vector< XType* >();
        }

        ObjectSet* tObjectSet = tTagMapIter->second;
        vector< XType* > tCastObjectSet;
        ObjectSet::const_iterator tIt;
        for( tIt = tObjectSet->begin(); tIt != tObjectSet->end(); tIt++ )
        {
            Object* tObject = *(tIt);
            if( XType* tCastObject = dynamic_cast< XType* >( tObject ) )
            {
                tCastObjectSet.push_back( tCastObject );
            }
        }
        return tCastObjectSet;
    }

}

#endif
