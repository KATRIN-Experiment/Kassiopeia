/**
 * @file KGPlyFileSpace.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2022-11-24
 */

#ifndef KGPLYFILESPACE_HH_
#define KGPLYFILESPACE_HH_

#include "KGPlyFile.hh"
#include "KGWrappedSpace.hh"

namespace KGeoBag
{

typedef KGWrappedSpace<KGPlyFile> KGPlyFileSpace;

}

#endif
