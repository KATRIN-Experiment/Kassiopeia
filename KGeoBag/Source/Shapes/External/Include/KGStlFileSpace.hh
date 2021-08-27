/**
 * @file KGStlFileSpace.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#ifndef KGSTLFILESPACE_HH_
#define KGSTLFILESPACE_HH_

#include "KGStlFile.hh"
#include "KGWrappedSpace.hh"

namespace KGeoBag
{

typedef KGWrappedSpace<KGStlFile> KGStlFileSpace;

}

#endif
