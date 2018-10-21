#ifndef KSERIALIZER_DEF
#define KSERIALIZER_DEF

#include "KMetadataStreamer.hh"

namespace KEMField
{
  template <class DataStreamer>
  class KSerializer;

  template <class DataStreamer>
  class KSerializerBase
  {
  public:
    virtual ~KSerializerBase() {}
    KMetadataStreamer& GetMetadataStreamer() { return fMetadataStreamer; }
    DataStreamer& GetDataStreamer()          { return fDataStreamer; }

  protected:
    KMetadataStreamer fMetadataStreamer;
    DataStreamer      fDataStreamer;
  };

  template <typename Type, class DataStreamer>
  class KSerializerType : virtual public KSerializerBase<DataStreamer>
  {
  public:
    friend inline KSerializer<DataStreamer>& operator>>(KSerializerType<Type,DataStreamer> &d,Type &x)
    {
      d.KSerializerBase<DataStreamer>::GetMetadataStreamer() >> x;
      d.KSerializerBase<DataStreamer>::GetDataStreamer() >> x;
      return d.Self();
    }

    friend inline KSerializer<DataStreamer>& operator<<(KSerializerType<Type,DataStreamer> &d,const Type &x)
    {
      d.KSerializerBase<DataStreamer>::GetMetadataStreamer() << x;
      d.KSerializerBase<DataStreamer>::GetDataStreamer() << x;
      return d.Self();
    }

    virtual ~KSerializerType() {}
    virtual KSerializer<DataStreamer>& Self() = 0;
  };

  template <class DataStreamer>
  class KSerializer :
    public KGenScatterHierarchyWithParameter<KEMField::FundamentalTypes,
					     DataStreamer,
					     KSerializerType >
  {
  public:
    KSerializer() {}
    virtual ~KSerializer() {}

    using KSerializerBase<DataStreamer>::GetMetadataStreamer;
    using KSerializerBase<DataStreamer>::GetDataStreamer;

    void open(const std::string& fileName,
	      const std::string& action="overwrite");
    void close();

    template <class Streamed>
    void PreStreamInAction(Streamed& s);
    template <class Streamed>
    void PostStreamInAction(Streamed& s);
    template <class Streamed>
    void PreStreamOutAction(const Streamed& s);
    template <class Streamed>
    void PostStreamOutAction(const Streamed& s);

  protected:
    KSerializer<DataStreamer>& Self() { return *this; }
    std::string fFileName;
  };

  template <class DataStreamer>
  void KSerializer<DataStreamer>::open(const std::string& fileName,
				       const std::string& action)
  {
    std::stringstream s;
    s << fileName << GetMetadataStreamer().GetFileSuffix();
    GetMetadataStreamer().open(s.str(),action);
    s.clear();s.str("");
    s << fileName << GetDataStreamer().GetFileSuffix();
    GetDataStreamer().open(s.str(),action);
  }

  template <class DataStreamer>
  void KSerializer<DataStreamer>::close()
  {
    GetMetadataStreamer().close();
    GetDataStreamer().close();
  }

  template <class DataStreamer>
  template <class Streamed>
  void KSerializer<DataStreamer>::PreStreamInAction(Streamed& s)
  {
    GetMetadataStreamer().PreStreamInAction(s);
    GetDataStreamer().PreStreamInAction(s);
  }

  template <class DataStreamer>
  template <class Streamed>
  void KSerializer<DataStreamer>::PostStreamInAction(Streamed& s)
  {
    GetMetadataStreamer().PostStreamInAction(s);
    GetDataStreamer().PostStreamInAction(s);
  }

  template <class DataStreamer>
  template <class Streamed>
  void KSerializer<DataStreamer>::PreStreamOutAction(const Streamed& s)
  {
    GetMetadataStreamer().PreStreamOutAction(s);
    GetDataStreamer().PreStreamOutAction(s);
  }

  template <class DataStreamer>
  template <class Streamed>
  void KSerializer<DataStreamer>::PostStreamOutAction(const Streamed& s)
  {
    GetMetadataStreamer().PostStreamOutAction(s);
    GetDataStreamer().PostStreamOutAction(s);
  }

}

#endif /* KSERIALIZER_DEF */
