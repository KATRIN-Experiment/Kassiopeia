#ifndef KBINARYDATASTREAMER_DEF
#define KBINARYDATASTREAMER_DEF

#include <fstream>
#include <string>

#include <iostream>

#include "KFundamentalTypes.hh"

namespace KEMField
{

/**
* @class KBinaryDataStreamer
*
* @brief A streamer class for raw binary I/O. 
*
* KBinaryDataStreamer is a class for streaming raw binary data.
*
* @author T.J. Corona
*/

  class KBinaryDataStreamer;

  template <typename Type>
  struct KBinaryDataStreamerType
  {
    friend inline KBinaryDataStreamer& operator>>(KBinaryDataStreamerType<Type>& d, Type &x)
    {
      d.Stream().read(reinterpret_cast<char*>(&x),sizeof(Type));
      return d.Self();
    }

    friend inline KBinaryDataStreamer &operator<<(KBinaryDataStreamerType<Type>& d, const Type &x)
    {
      d.Stream().write(reinterpret_cast<const char*>(&x),sizeof(Type));
      return d.Self();
    }
    virtual ~KBinaryDataStreamerType() {}
    virtual std::fstream& Stream() = 0;
    virtual KBinaryDataStreamer& Self() = 0;
  };

  template <>
  struct KBinaryDataStreamerType<std::string>
  {
    friend inline KBinaryDataStreamer& operator>>(KBinaryDataStreamerType<std::string>& d, std::string &x)
    {
      unsigned int size;
      d.Stream().read(reinterpret_cast<char*>(&size),sizeof(unsigned int));
      x.resize(size);
      d.Stream().read(&x[0],size);
      return d.Self();
    }

    friend inline KBinaryDataStreamer &operator<<(KBinaryDataStreamerType<std::string>& d, const std::string &x)
    {
      const unsigned int size = x.size();
      d.Stream().write(reinterpret_cast<const char*>(&size),sizeof(unsigned int));
      d.Stream().write(x.c_str(),size);
      return d.Self();
    }
    virtual ~KBinaryDataStreamerType() {}
    virtual std::fstream& Stream() = 0;
    virtual KBinaryDataStreamer& Self() = 0;
  };

  typedef KGenScatterHierarchy<KEMField::FundamentalTypes,
			       KBinaryDataStreamerType>
  KBinaryDataStreamerFundamentalTypes;

  class KBinaryDataStreamer : public KBinaryDataStreamerFundamentalTypes
  {
  public:
    KBinaryDataStreamer() {}
    virtual ~KBinaryDataStreamer() {}

    void open(const std::string& fileName,const std::string& action="update");
    void close() { fFile.close(); }

    template <class Streamed>
    void PreStreamInAction(Streamed&) {}
    template <class Streamed>
    void PostStreamInAction(Streamed&) {}
    template <class Streamed>
    void PreStreamOutAction(const Streamed&) {}
    template <class Streamed>
    void PostStreamOutAction(const Streamed&) {}

    std::string GetFileSuffix() const { return ".kbd"; }

    std::fstream& Stream() { return fFile; }
    const std::fstream& Stream() const { return fFile; }

  protected:
    KBinaryDataStreamer& Self() { return *this; }

    std::fstream fFile;
  };
}

#endif /* KBINARYDATASTREAMER_DEF */
