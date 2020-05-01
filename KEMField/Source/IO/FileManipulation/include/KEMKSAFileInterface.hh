#ifndef __KEMKSAFileInterface_H__
#define __KEMKSAFileInterface_H__

#include "KEMFile.hh"
#include "KEMFileInterface.hh"
#include "KSAInputNode.hh"
#include "KSAOutputNode.hh"


namespace KEMField
{

/**
*
*@file KEMKSAFileInterface.hh
*@class KEMKSAFileInterface
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov  5 23:41:46 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KEMKSAFileInterface
{
  public:
    KEMKSAFileInterface(){};
    virtual ~KEMKSAFileInterface(){};

    static void ReadKSAFile(KSAInputNode* node, string file_name, bool& result);
    static void SaveKSAFile(KSAOutputNode* node, string file_name, bool& result, bool forceOverwrite = false);

  protected:
    /* data */
};


}  // namespace KEMField


#endif /* __KEMKSAFileInterface_H__ */
