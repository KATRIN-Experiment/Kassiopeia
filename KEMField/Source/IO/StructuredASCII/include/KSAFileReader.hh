#ifndef KSAFileReader_HH__
#define KSAFileReader_HH__


#include "KSADefinitions.hh"

#include <fstream>
#include <queue>
#include <string>

#ifndef KEMFIELD_USE_ZLIB
#include "miniz.hh"
#else
#include "zlib.h"
#endif

#define EXPAN 64

namespace KEMField
{

/**
*
*@file KSAFileReader.hh
*@class KSAFileReader
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Dec 17 14:56:07 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAFileReader
{
  public:
    KSAFileReader();
    virtual ~KSAFileReader();

    void SetFileName(const std::string& filename);
    bool Open();
    void Close();

    bool GetLine(std::string& line);

  protected:
    void ExtractData();
    void ExtractLines();
    std::string StripWhiteSpace();

    bool fIsOpen;
    bool fIsFinished;
    std::string fFileName;
    std::string fLine;
    std::string fLineBuffer;
    std::ifstream fFileStream;

    bool fUseDecompression;
    z_stream fZStream;


    unsigned char* in_buffer;   //[READ_CHUNK];
    unsigned char* out_buffer;  //[EXPAN*READ_CHUNK];

    int fUsedSpace;

    //temporary storage buffers
    std::vector<unsigned char> fInputBuffer;
    std::vector<unsigned char> fOutputBuffer;
    std::vector<unsigned char> fLineStagingBuffer;

    std::queue<std::string> fLineQueue;

    int e_count;
};


}  // namespace KEMField

#endif /* KSAFileReader_H__ */
