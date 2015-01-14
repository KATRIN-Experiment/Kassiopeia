#ifndef KGAREA_HH_
#define KGAREA_HH_

#include "KTwoVector.hh"
#include "KThreeVector.hh"
#include "KTransformation.hh"

#include "KGCoreMessage.hh"
#include "KGVisitor.hh"

#include "KTagged.h"
using katrin::KTagged;

#include "KConst.h"
using katrin::KConst;

#include "KSmartPointer.h"
using katrin::KSmartPointer;

#include <cmath>

namespace KGeoBag
{

    class KGArea :
        public KTagged
    {
    	public:
    		class Visitor {
    		public:
    			Visitor() {}
				virtual ~Visitor() {}
				virtual void VisitArea(KGArea*) = 0;
    		};

        public:
            KGArea();
            KGArea( const KGArea& aArea );
            virtual ~KGArea();

        public:
            void Accept( KGVisitor* aVisitor );

        protected:
            virtual void AreaAccept( KGVisitor* aVisitor );

        public:
            bool Above( const KThreeVector& aPoint ) const;
            KThreeVector Point( const KThreeVector& aPoint ) const;
            KThreeVector Normal( const KThreeVector& aPoint ) const;

        protected:
            virtual bool AreaAbove( const KThreeVector& aPoint ) const = 0;
            virtual KThreeVector AreaPoint( const KThreeVector& aPoint ) const = 0;
            virtual KThreeVector AreaNormal( const KThreeVector& aPoint ) const = 0;

        protected:
            void Check() const;
            virtual void AreaInitialize() const = 0;
            mutable bool fInitialized;
    };

}

#endif
