#ifndef KELECTROMAGNETCONTAINER_H
#define KELECTROMAGNETCONTAINER_H

#include <vector>

#include "KTypeManipulation.hh"
#include "KElectromagnetTypes.hh"

namespace KEMField
{
  template <class Electromagnet>
  class KElectromagnetContainerType : public std::vector<Electromagnet*>
  {
  public:
    KElectromagnetContainerType() {}
    virtual ~KElectromagnetContainerType() {}

    typedef typename std::vector<Electromagnet*>::iterator ElectromagnetIterator;
    typedef typename std::vector<Electromagnet*>::const_iterator ElectromagnetCIterator;

    template <class Stream>
    friend Stream& operator>>(Stream& s,KElectromagnetContainerType<Electromagnet>& c)
    {
      c.Clear();
      unsigned int nElectromagnets;
      s >> nElectromagnets;
      for (unsigned int i=0;i<nElectromagnets;i++)
      {
	Electromagnet* e = new Electromagnet();
	s >> e;
	c.push_back(e);
      }
      return s;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& s,const KElectromagnetContainerType<Electromagnet>& c)
    {
      s << c.size();
      for (ElectromagnetCIterator it = c.begin(); it != c.end(); ++it)
	s << *(*it);
      return s;
    }

    virtual bool IsOwner() const = 0;

    void Clear();
  };

  template <class Electromagnet>
  void KElectromagnetContainerType<Electromagnet>::Clear()
  {
    if (IsOwner())
    {
      for (ElectromagnetIterator it=std::vector<Electromagnet*>::begin();it!=std::vector<Electromagnet*>::end();++it)
	delete *it;
    }
    std::vector<Electromagnet*>::clear();
  }

  class KElectromagnetContainer;

  template <int typeID=0>
  class KElectromagnetAction
  {
  public:
    typedef typename KEMField::TypeAt<KEMField::KElectromagnetTypes,typeID>::Result Electromagnet;
    typedef KElectromagnetContainerType<Electromagnet> ElectromagnetContainer;

    static void push_back(KElectromagnetContainer& c,KElectromagnet* e);
    static unsigned int size(const KElectromagnetContainer& c);
    static KElectromagnet* at(unsigned int i,KElectromagnetContainer& c);
    static const KElectromagnet* at(unsigned int i, const KElectromagnetContainer& c);
    static void Clear(KElectromagnetContainer& c);

    template <class Action>
    static void ActOnElectromagnets(Action& action);

    template <class Stream>
    static Stream& StreamIn(Stream& stream, KElectromagnetContainer& c)
    {
      ElectromagnetContainer& container = static_cast<ElectromagnetContainer>(c);
      stream >> container;
      return KElectromagnetAction<typeID+1>::StreamIn(stream,c);
    }

    template <class Stream>
    static Stream& StreamOut(Stream& stream, const KElectromagnetContainer& c)
    {
      const ElectromagnetContainer& container = static_cast<const ElectromagnetContainer&>(c);
      stream << container;
      return KElectromagnetAction<typeID+1>::StreamOut(stream,c);
    }

  };

  typedef KGenScatterHierarchy<KEMField::KElectromagnetTypes,KElectromagnetContainerType> KElectromagnetContainerTypes;

  class KElectromagnetContainer : public KElectromagnetContainerTypes
  {
  public:
    KElectromagnetContainer() : fIsOwner(true) {}
    virtual ~KElectromagnetContainer() { clear(); }

    static std::string Name() { return "ElectromagnetContainer"; }

    template <class Electromagnet>
    void push_back(Electromagnet* e) { KElectromagnetContainerType<Electromagnet>::push_back(e); }

    template <class Electromagnet>
    const KElectromagnetContainerType<Electromagnet>& Vector() const { return static_cast<const KElectromagnetContainerType<Electromagnet>&>(*this); }

    template <class Electromagnet>
    KElectromagnetContainerType<Electromagnet>& Vector() { return static_cast<KElectromagnetContainerType<Electromagnet>&>(*this); }

    void push_back(KElectromagnet* e) { KElectromagnetAction<>::push_back(*this,e); }
    unsigned int size() const { return KElectromagnetAction<>::size(*this); }
    bool empty() const { return KElectromagnetAction<>::size(*this) == 0; }
    void clear() { return KElectromagnetAction<>::Clear(*this); }

    KElectromagnet* at(unsigned int i) { return KElectromagnetAction<>::at(i, *this); }
    KElectromagnet* operator[](unsigned int i) { return at(i); }

    const KElectromagnet* at(unsigned int i) const { return KElectromagnetAction<>::at(i, *this); }
    const KElectromagnet* operator[](unsigned int i) const { return at(i); }

    template <class Electromagnet>
    unsigned int size() const { return KElectromagnetContainerType<Electromagnet>::size(); }

    template <class Electromagnet>
    Electromagnet* at(unsigned int i) { return KElectromagnetContainerType<Electromagnet>::at(i); }

    template <class Electromagnet>
    const Electromagnet* at(unsigned int i) const { return KElectromagnetContainerType<Electromagnet>::at(i); }

    void IsOwner(bool choice) { fIsOwner = choice; }
    bool IsOwner() const { return fIsOwner; }

  protected:
    bool fIsOwner;
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KElectromagnetContainer& c)
  {
    s.PreStreamInAction(c);
    KElectromagnetAction<>::StreamIn(s,c);
    s.PostStreamInAction(c);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KElectromagnetContainer& c)
  {
    s.PreStreamOutAction(c);
    KElectromagnetAction<>::StreamOut(s,c);
    s.PostStreamOutAction(c);
    return s;
  }

  template <int typeID>
  void KElectromagnetAction<typeID>::push_back(KElectromagnetContainer& c,KElectromagnet* e)
  {
    if (Electromagnet* m = dynamic_cast<Electromagnet*>(e))
      c.ElectromagnetContainer::push_back(m);
    else
      return KElectromagnetAction<typeID+1>::push_back(c,e);
  }

  template <int typeID>
  unsigned int KElectromagnetAction<typeID>::size(const KElectromagnetContainer& c)
  {
    return c.ElectromagnetContainer::size() + KElectromagnetAction<typeID+1>::size(c);
  }

  template <int typeID>
  void KElectromagnetAction<typeID>::Clear(KElectromagnetContainer& c)
  {
    c.ElectromagnetContainer::Clear();
    return KElectromagnetAction<typeID+1>::Clear(c);
  }

  template <int typeID>
  KElectromagnet* KElectromagnetAction<typeID>::at(unsigned int i,KElectromagnetContainer& c)
  {
    if (i < c.ElectromagnetContainer::size())
      return c.ElectromagnetContainer::at(i);
    return KElectromagnetAction<typeID+1>::at(i-c.ElectromagnetContainer::size(),c);
  }

  template <int typeID>
  const KElectromagnet* KElectromagnetAction<typeID>::at(unsigned int i,const KElectromagnetContainer& c)
  {
    if (c.ElectromagnetContainer::size() < i)
      return c.ElectromagnetContainer::at(i);
    return KElectromagnetAction<typeID+1>::at(i-c.ElectromagnetContainer::size(),c);
  }

  template <int typeID>
  template <class Action>
  void KElectromagnetAction<typeID>::ActOnElectromagnets(Action& action)
  {
    action.Act(Type2Type<Electromagnet>());
    return KElectromagnetAction<typeID+1>::template ActOnElectromagnets<Action>(action);
  }

  template <>
  class KElectromagnetAction<Length<KEMField::KElectromagnetTypes>::value>
  {
  public:
    static void push_back(KElectromagnetContainer&,KElectromagnet*) {}
    static unsigned int size(const KElectromagnetContainer&) { return 0; }
    static KElectromagnet* at(unsigned int,KElectromagnetContainer&) { return NULL; }
    static const KElectromagnet* at(unsigned int,const KElectromagnetContainer&) { return NULL; }
    static void Clear(KElectromagnetContainer&) {}
    template <class Action>
    static void ActOnElectromagnets(Action&) {}
    template <class Stream>
    static Stream& StreamIn(Stream& stream, KElectromagnetContainer&) { return stream; }
    template <class Stream>
    static Stream& StreamOut(Stream& stream, const KElectromagnetContainer&) { return stream; }
  };

}

#endif /* KELECTROMAGNETCONTAINER */
