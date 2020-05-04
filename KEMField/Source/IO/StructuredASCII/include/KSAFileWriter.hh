#ifndef KSAFileWriter_HH__
#define KSAFileWriter_HH__

#include "KSADefinitions.hh"

#include <fstream>
#include <string>

#ifndef KEMFIELD_USE_ZLIB
#include "miniz.hh"
#else
#include "zlib.h"
#endif


namespace KEMField
{

/**
*
*@file KSAFileWriter.hh
*@class KSAFileWriter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec 18 09:37:32 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAFileWriter
{
  public:
    KSAFileWriter();
    virtual ~KSAFileWriter();

    //must be called before opening the file stream
    void SetFileName(std::string filename);

    bool Open();
    void AddToFile(const std::string& data);
    void FlushStream()
    {
        fFileStream.flush();
    }
    void Close();

    std::string GetFileName() const
    {
        return fFileName;
    }

  protected:
    bool fIsOpen;

    bool fUseCompression;
    int fCompressionLevel;
    z_stream fZStream;
    unsigned char* in_buffer;   //[WRITE_CHUNK];
    unsigned char* out_buffer;  //[WRITE_CHUNK];
    int fUsedSpace;


    std::string fFileName;
    std::ofstream fFileStream;
};


}  // namespace KEMField

#endif /* KSAFileWriter_H__ */
