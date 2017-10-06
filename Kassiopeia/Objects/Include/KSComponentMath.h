#ifndef Kassiopeia_KSComponentMath_h_
#define Kassiopeia_KSComponentMath_h_

#include "KSDictionary.h"
#include "KSNumerical.h"
#include "TF1.h"
#include <memory>

namespace Kassiopeia
{

    template< class XValueType >
    class KSComponentMath :
        public KSComponent
    {
        public:
            KSComponentMath( std::vector< KSComponent* > aParentComponents, std::vector< XValueType* > aParentPointers, std::string aTerm ) :
                    KSComponent(),
                    fParentComponents( aParentComponents ),
                    fParents( aParentPointers ),
                    fResult( KSNumerical< XValueType >::Zero() )
            {
                Set( &fResult );
                this->SetParent( aParentComponents.at( 0 ) );
                for( size_t tIndex = 0; tIndex < aParentComponents.size(); tIndex++ )
                {
                    aParentComponents.at( tIndex )->AddChild( this );
                }

                for( size_t tIndex = 0; tIndex < fParents.size(); tIndex++ )
                {
                    //create std::string for variable name x0,x1,etc.
                    std::string tVariable( "x" );
                    std::stringstream tIndexConverter;
                    tIndexConverter << tIndex;
                    tVariable += tIndexConverter.str();

                    std::stringstream tParameterConverter;
                    tParameterConverter << "[" << tIndexConverter.str() << "]";

                    //replace x with [index], this denotes a parameter for TF1
                    while( aTerm.find( tVariable ) != std::string::npos )
                    {
                        aTerm.replace( aTerm.find( tVariable ), tVariable.length(), tParameterConverter.str() );
                    }
                }

                fTerm = aTerm;

                //check if all x are replaced in formula
                if( fTerm.find( std::string( "x" ) ) != std::string::npos || fTerm.find( std::string( "X" ) ) != std::string::npos )
                {
                    objctmsg( eError ) << "Error in KSComponentMath: could not replace all variables in term! Use only x0,x1,etc., one for each component" << eom;
                }

                // initialize function once, parameters are updated every PushUpdate call
                fFunction = std::make_shared<TF1>( "(anonymous)", fTerm.c_str(), -1., 1. );
            }
            KSComponentMath( const KSComponentMath< XValueType >& aCopy ) :
                    KSComponent( aCopy ),
                    fParentComponents( aCopy.fParentComponents ),
                    fParents( aCopy.fParents ),
                    fResult( aCopy.fResult ),
                    fTerm( aCopy.fTerm ),
                    fFunction( aCopy.fFunction )
            {
                Set( &fResult );
                this->SetParent( aCopy.fParentComponent );
                for( size_t tIndex = 0; tIndex < aCopy.fParentComponents.size(); tIndex++ )
                {
                    aCopy.fParentComponents.at( tIndex )->AddChild( this );
                }
            }
            virtual ~KSComponentMath()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Clone() const
            {
                return new KSComponentMath< XValueType >( *this );
            }
            KSComponent* Component( const std::string& aField )
            {
                objctmsg_debug( "component math <" << this->GetName() << "> building component named <" << aField << ">" << eom )
                KSComponent* tComponent = KSDictionary< XValueType >::GetComponent( this, aField );
                if( tComponent == NULL )
                {
                    objctmsg( eError ) << "component math <" << this->GetName() << "> has no output named <" << aField << ">" << eom;
                }
                else
                {
                    fChildComponents.push_back( tComponent );
                }
                return tComponent;
            }
            KSCommand* Command( const std::string& /*aField*/, KSComponent* /*aChild*/)
            {
                return NULL;
            }

        public:
            void PushUpdateComponent()
            {
                objctmsg_debug( "component math <" << this->GetName() << "> pushing update" << eom );

                for( size_t tIndex = 0; tIndex < fParents.size(); tIndex++ )
                {
                    fParentComponents.at( tIndex )->PullUpdate();
                    fFunction->SetParameter(tIndex, *(fParents.at(tIndex)));
                }

                fResult = fFunction->Eval( 0 );
                return;
            }

//            void PullUpdateComponent()
//            {
//                objctmsg_debug( "component math <" << this->GetName() << "> pulling Update" << eom );
//                for ( size_t tIndex = 1; tIndex < fParentComponents.size(); tIndex++ )
//                {
//                	fParentComponents.at( tIndex )->PullUpdate();
//                }
//                return;
//            }
//
//            void PullDeupdateComponent()
//            {
//                objctmsg_debug( "component math <" << this->GetName() << "> pulling deupdate" << eom );
//                for ( size_t tIndex = 1; tIndex < fParentComponents.size(); tIndex++ )
//                {
//                	fParentComponents.at( tIndex )->PullDeupdate();
//                }
//                return;
//            }

        private:
            std::vector< KSComponent* > fParentComponents;
            std::vector< XValueType* > fParents;
            XValueType fResult;
            std::string fTerm;
            std::shared_ptr<TF1> fFunction;
    };

}

#endif
