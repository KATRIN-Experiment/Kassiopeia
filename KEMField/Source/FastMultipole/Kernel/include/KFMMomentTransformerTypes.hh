#ifndef KFMMomentTransformerTypes_HH__
#define KFMMomentTransformerTypes_HH__

#include "KFMMomentTransformer.hh"

#include "KFMResponseKernel_3DLaplaceL2L.hh"
#include "KFMResponseKernel_3DLaplaceM2L.hh"
#include "KFMResponseKernel_3DLaplaceM2M.hh"

namespace KEMField
{

/*
*
*@file KFMMomentTransformerTypes.hh
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec  3 15:40:41 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

typedef KFMMomentTransformer<KFMResponseKernel_3DLaplaceM2M> KFMMomentTransformer_3DLaplaceM2M;
typedef KFMMomentTransformer<KFMResponseKernel_3DLaplaceM2L> KFMMomentTransformer_3DLaplaceM2L;
typedef KFMMomentTransformer<KFMResponseKernel_3DLaplaceL2L> KFMMomentTransformer_3DLaplaceL2L;

}

#endif /* KFMMomentTransformerTypes_H__ */
