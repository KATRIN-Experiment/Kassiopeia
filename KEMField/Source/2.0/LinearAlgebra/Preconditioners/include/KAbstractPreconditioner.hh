#ifndef KAbstractPreconditioner_HH__
#define KAbstractPreconditioner_HH__

#include "KSquareMatrix.hh"

namespace KEMField
{


/*
*
*@file KAbstractPreconditioner.hh
*@class KAbstractPreconditioner
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jun  3 09:42:09 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template < typename ValueType >
class KAbstractPreconditioner: public KSquareMatrix<ValueType>
{
    public:

        KAbstractPreconditioner(){};
        ~KAbstractPreconditioner(){};

};


}//end of namespace

#endif /* KAbstractPreconditioner_H__ */
