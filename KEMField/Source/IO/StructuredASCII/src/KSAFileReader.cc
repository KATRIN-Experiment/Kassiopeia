#include "KSAFileReader.hh"


#include <iostream>

#include <assert.h>

namespace KEMField{


KSAFileReader::KSAFileReader():fIsOpen(false),fUseDecompression(false)
{
    fUsedSpace = 0;
    fIsFinished = false;
    fLineBuffer = "";
    e_count = 0;

//    in_buffer = new unsigned char[READ_CHUNK];
//    out_buffer = new unsigned char[EXPAN*READ_CHUNK];

    fInputBuffer.reserve(READ_CHUNK);
    fOutputBuffer.reserve(EXPAN*READ_CHUNK);
    fLineStagingBuffer.reserve(EXPAN*READ_CHUNK);
    in_buffer = &(fInputBuffer[0]);
    out_buffer = &(fOutputBuffer[0]);

}

KSAFileReader::~KSAFileReader()
{
//    delete[] in_buffer;
//    delete[] out_buffer;
}

void KSAFileReader::SetFileName(std::string filename)
{
    fFileName = filename;

    if(std::string::npos != filename.find(std::string(".ksa")))
    {
        fUseDecompression = false; //plain text
    }
    else
    {
        //default
        fUseDecompression = false; //plain text
    }

    //if we used the special extention we'll compress things
    if(std::string::npos != filename.find(std::string(".zksa")))
    {
        fUseDecompression = true; //compressed text
    }


}


bool
KSAFileReader::Open()
{
    if(fUseDecompression)
    {
        fLineBuffer = "";
        /* allocate inflate state */
        int ret;
        fZStream.zalloc = Z_NULL;
        fZStream.zfree = Z_NULL;
        fZStream.opaque = Z_NULL;
        fZStream.avail_in = 0;
        fZStream.next_in = Z_NULL;
        ret = inflateInit(&fZStream);
        if(ret != Z_OK){return false;};
        fUsedSpace = 0;

        if(fIsOpen)
        {
            fFileStream.close();
        }

        fFileStream.open(fFileName.c_str(), std::ifstream::in | std::ifstream::binary);
        fIsOpen = fFileStream.is_open();
        fIsFinished = false;
        return fIsOpen;
    }
    else
    {
        if(fIsOpen)
        {
            fFileStream.close();
        }

        fFileStream.open(fFileName.c_str(), std::ifstream::in);
        fIsOpen = fFileStream.is_open();
        fIsFinished = false;
        return fIsOpen;
    }

}


bool
KSAFileReader::GetLine(std::string& line)
{
    if(fUseDecompression)
    {
        if(fLineQueue.size() > 0)
        {
            fLine = fLineQueue.front();
            line = StripWhiteSpace();
            fLineQueue.pop();
            return true;
        }
        else if(fLineQueue.size() == 0 && !fIsFinished)
        {
            while( (fLineQueue.size() < 100 ) && !fIsFinished )
            {
                ExtractData();
            }

            if(fLineQueue.size() > 0)
            {
                fLine = fLineQueue.front();
                line = StripWhiteSpace();
                fLineQueue.pop();
                return true;
            }
            else
            {
                line = "";
                return false;
            }
        }
        else
        {
            line = "";
            return false;
        }
    }
    else
    {
        if(fFileStream.good())
        {
            std::getline(fFileStream, fLine); //get the line

            //strip leading and trailing whitespace
            line = StripWhiteSpace();
            return true;
        }

        line = "";
        return false;
    }
}


void KSAFileReader::ExtractData()
{
    std::streamsize length = READ_CHUNK;
    std::streamsize n_retrieved;
    unsigned int have;
    int ret;
    int flush;

    if(fFileStream.good() && !fIsFinished)
    {
        e_count++;
        fFileStream.read(reinterpret_cast<char*>(in_buffer), length);
        n_retrieved = fFileStream.gcount();

        if(fFileStream.eof())
        {
            fIsFinished = true;
        }; //end of file reached

        if(fFileStream.bad())
        {
            //ERROR!
            fIsFinished = true;
        }

        if(n_retrieved != 0)
        {

            if(!fIsFinished)
            {
                flush = Z_NO_FLUSH;
            }
            else
            {
                flush = Z_FINISH;
            }


            fZStream.avail_in = n_retrieved;
            fZStream.next_in = in_buffer;

            /* run inflate() on input until output buffer not full */
            do
            {
                fZStream.avail_out = EXPAN*READ_CHUNK;
                fZStream.next_out = out_buffer;
                ret = inflate(&fZStream, flush);
                assert(ret != Z_STREAM_ERROR);  /* state not clobbered */

                switch (ret)
                {
                    case Z_NEED_DICT: ret = Z_DATA_ERROR;     /* and fall through, use attribute [[fallthrough]] for newer compilers */
						#if defined(__GNUC__) && (__GNUC__ >= 7)
							[[fallthrough]];
						#endif
                    case Z_DATA_ERROR:
						#if defined(__GNUC__) && (__GNUC__ >= 7)
							[[fallthrough]];
						#endif
                    case Z_MEM_ERROR: (void)inflateEnd(&fZStream); fIsFinished = true; break;
                };

                have = EXPAN*READ_CHUNK - fZStream.avail_out;

                //std::cout<<"# of bytes we have : "<<have<<std::endl;

                //copy into the line staging buffer
                fLineStagingBuffer.insert(fLineStagingBuffer.end(), fOutputBuffer.begin(), fOutputBuffer.begin() + have);

//                for(unsigned int i=0; i < have; i++)
//                {
//                    fCharacterBuffer.push_back(out_buffer[i]);
//                }

                ExtractLines();

            }while(fZStream.avail_out == 0);

        }

        if( fIsFinished )
        {
            (void)inflateEnd(&fZStream);
        }
    }

}

void KSAFileReader::ExtractLines()
{
//    if(fCharacterQueue.size() != 0)
//    {
//        do
//        {
//            if(fCharacterQueue.front() == '\n')
//            {
//                fLineQueue.push(fLineBuffer);

////                std::string end_of_line;
////                if(fLineBuffer.size() > 100 )
////                {
////                    end_of_line = fLineBuffer.substr(fLineBuffer.size()-101, fLineBuffer.size()-1);
////                }
////                else
////                {
////                    end_of_line = fLineBuffer;
////                }


////                std::cout<<"line buffer: "<<end_of_line<<std::endl;
//                fLineBuffer = "";
//            }
//            else
//            {
//                fLineBuffer.push_back(fCharacterQueue.front());
//            }

//            fCharacterQueue.pop();
//        }
//        while( fCharacterQueue.size() != 0 );
//    }

    std::vector<unsigned char>::iterator line_begin = fLineStagingBuffer.begin();
    std::vector<unsigned char>::iterator it;

    for( it = fLineStagingBuffer.begin(); it != fLineStagingBuffer.end();)
    {
        if(*it == '\n')
        {
            fLineBuffer.insert(fLineBuffer.begin(), line_begin, it);
            fLineQueue.push(fLineBuffer);
            fLineBuffer = "";
            ++it;
            line_begin = it;
        }
        else
        {
            ++it;
        }
    }

    fLineStagingBuffer.erase(fLineStagingBuffer.begin(), line_begin);
}



void KSAFileReader::Close()
{
    if(fUseDecompression)
    {
        if(fIsOpen)
        {
            fFileStream.close();
        }
    }
    else
    {
        if(fIsOpen)
        {
            fFileStream.close();
        }
    }
}

std::string KSAFileReader::StripWhiteSpace()
{
    size_t begin;
    size_t end;
    size_t len;

    begin = fLine.find_first_not_of(" \t");

    if(begin != std::string::npos)
    {
        end = fLine.find_last_not_of(" \t");

        len = end - begin + 1;

        return fLine.substr(begin, len);
    }

    //empty string
    return "";
}




}//end of kemfield namespace
