
#ifndef KSURFACE_DEF
#error "Do not include KSurfaceAction.hh directly. Include KSurface.hh instead."
#endif

#ifndef KSURFACEACTION_DEF
#define KSURFACEACTION_DEF

#include "KTypeManipulation.hh"
#include "KFundamentalTypeCounter.hh"

#include "../../../Surfaces/include/KSurfaceID.hh"
#include "../../../Surfaces/include/KSurfaceTypes.hh"

namespace KEMField
{

/**
* @class KShapesAction
*
* @brief A class template for acting on all surfaces, each by shape policy.
*
* KShapesAction is a template for performing an action on each surface according
* to each shape type.
*
* @author T.J. Corona
*/

  template <class Action,int shapeID=0>
  class KShapesAction
  {
  public:
    static void ActOnShapes(Action& anAction)
    {
      anAction.PerformAction(Type2Type<typename TypeAt< KShapeTypes, shapeID>::Result >());
      return KShapesAction<Action,shapeID+1>::ActOnShapes(anAction);
    }
  };

  template <class Action>
  class KShapesAction<Action,Length<KShapeTypes>::value>
  {
  public:
    static void ActOnShapes(Action&) {}
  };

/**
* @class KBoundariesAction
*
* @brief A class template for acting on all surfaces, each by boundary policy.
*
* KBoundariesAction is a template for performing an action on each surface
* according to each boundary type.
*
* @author T.J. Corona
*/

  template <class Action,int boundaryID=0>
  class KBoundariesAction
  {
  public:
    static void ActOnBoundaries(Action& anAction)
    {
      anAction.PerformAction(Type2Type<typename TypeAt< KBoundaryTypes, boundaryID>::Result >());
      return KBoundariesAction<Action,boundaryID+1>::ActOnBoundaries(anAction);
    }
  };

  template <class Action>
  class KBoundariesAction<Action,Length<KBoundaryTypes>::value>
  {
  public:
    static void ActOnBoundaries(Action&) {}
  };

/**
* @class KBasesAction
*
* @brief A class template for acting on all surfaces, each by basis policy.
*
* KBasesAction is a template for performing an action on each surface
* according to each basis type.
*
* @author T.J. Corona
*/

  template <class Action,int basisID=0>
  class KBasesAction
  {
  public:
    static void ActOnBases(Action& anAction)
    {
      anAction.PerformAction(Type2Type<typename TypeAt< KBasisTypes, basisID>::Result >());
      return KBasesAction<Action,basisID+1>::ActOnBases(anAction);
    }
  };

  template <class Action>
  class KBasesAction<Action,Length<KBasisTypes>::value>
  {
  public:
    static void ActOnBases(Action&) {}
  };

/**
* @class KShapeAction
*
* @brief A class template for dispatch on surfaces by shape policy.
*
* KShapeAction is a template for facilitating dispatch on surfaces according
* to Shape policy.
*
* @author T.J. Corona
*/

  template <class Action,int shapeID=0>
  class KShapeAction
  {
  public:
    static void ActOnShapeType(const KSurfaceID& anID,Action& anAction)
    {
      if (anID.ShapeID == shapeID)
      {
	anAction.PerformAction(Type2Type<typename TypeAt< KShapeTypes, shapeID >::Result >());
	return;
      }
      else
	return KShapeAction<Action,shapeID+1>::ActOnShapeType(anID,anAction);
    }
  };

  template <class Action>
  class KShapeAction<Action,Length<KShapeTypes>::value>
  {
  public:
    static void ActOnShapeType(const KSurfaceID&,Action&) {}
  };

/**
* @class KBoundaryAction
*
* @brief A class template for dispatch on surfaces by boundary policy.
*
* KBoundaryAction is a template for facilitating dispatch on surfaces according
* to Boundary policy.
*
* @author T.J. Corona
*/

  template <class Action,int basisID=0,int boundaryID=0>
  class KBoundaryAction
  {
  public:
    static void ActOnBoundaryType(const KSurfaceID& anID,Action& anAction)
    {
      if (anID.BasisID == basisID)
      {
	if (anID.BoundaryID == boundaryID)
	{
	  anAction.PerformAction(Type2Type < KBoundaryType<typename TypeAt< KBasisTypes, basisID >::Result,typename TypeAt< KBoundaryTypes, boundaryID >::Result > > ());
	  return;
	}
	else
	  return KBoundaryAction<Action,basisID,boundaryID+1>::ActOnBoundaryType(anID,anAction);
      }
      else
	return KBoundaryAction<Action,basisID+1,boundaryID>::ActOnBoundaryType(anID,anAction);
    }
  };

  template <class Action,int basisID>
  class KBoundaryAction<Action,basisID,Length<KBoundaryTypes>::value>
  {
  public:
    static void ActOnBoundaryType(const KSurfaceID&,Action&) {}
  };

  template <class Action,int boundaryID>
  class KBoundaryAction<Action,Length<KBasisTypes>::value,boundaryID>
  {
  public:
    static void ActOnBoundaryType(const KSurfaceID&,Action&) {}
  };

/**
* @class KBasisAction
*
* @brief A class template for dispatch on surfaces by basis policy.
*
* KBasisAction is a template for facilitating dispatch on surfaces according
* to Basis policy.
*
* @author T.J. Corona
*/

  template <class Action,int basisID=0>
  class KBasisAction
  {
  public:
    static void ActOnBasisType(const KSurfaceID& anID,Action& anAction)
    {
      if (anID.BasisID == basisID)
      {
	anAction.PerformAction(Type2Type<typename TypeAt< KBasisTypes, basisID >::Result >());
	return;
      }
      else
	return KBasisAction<Action,basisID+1>::ActOnBasisType(anID,anAction);
    }
  };

  template <class Action>
  class KBasisAction<Action,Length<KBasisTypes>::value>
  {
  public:
    static void ActOnBasisType(const KSurfaceID&,Action&) {}
  };

/**
* @class KSurfaceAction
*
* @brief A class template for dispatch on surfaces by policy.
*
* KSurfaceAction is a template for facilitating dispatch on surfaces according
* to specific Basis, Boundary and Shape combinations.
*
* @author T.J. Corona
*/

  template <class Action,int basisID=0,int boundaryID=0,int shapeID=0>
  class KSurfaceAction
  {
  public:
    static void ActOnSurfaceType(const KSurfaceID& anID,Action& anAction)
    {
      if (anID.BasisID == basisID)
      {
  	if (anID.BoundaryID == boundaryID)
  	{
  	  if (anID.ShapeID == shapeID)
  	  {
  	    anAction.PerformAction(Type2Type<KSurface< typename TypeAt< KBasisTypes,basisID >::Result, typename TypeAt< KBoundaryTypes, boundaryID >::Result, typename TypeAt< KShapeTypes, shapeID >::Result > >());
  	    return;
      }
  	  else
  	    return KSurfaceAction<Action,
  				  basisID,
  				  boundaryID,
  				  shapeID+1>::ActOnSurfaceType(anID,anAction);
    }
  	else
  	  return KSurfaceAction<Action,
  				basisID,
  				boundaryID+1,
  				shapeID>::ActOnSurfaceType(anID,anAction);
  }
      else
  	return KSurfaceAction<Action,
  			      basisID+1,
  			      boundaryID,
  			      shapeID>::ActOnSurfaceType(anID,anAction);
    }
  };

  template <typename Action,int boundaryID,int basisID>
  class KSurfaceAction<Action,basisID,boundaryID,Length<KShapeTypes>::value>
  {
  public:
    static void ActOnSurfaceType(const KSurfaceID& anID,Action&)
    {
      std::cout<<"Unable to find a shape whose ID matches the ID in question ("<<anID.ShapeID<<")."<<std::endl;
      return;
    }
  };

  template <typename Action,int basisID,int shapeID>
  class KSurfaceAction<Action,basisID,Length<KBoundaryTypes>::value,shapeID>
  {
  public:
    static void ActOnSurfaceType(const KSurfaceID& anID,Action&)
    {
      std::cout<<"Unable to find a boundary whose ID matches the ID in question ("<<anID.BoundaryID<<")."<<std::endl;
      return;
    }
  };

  template <typename Action,int boundaryID,int shapeID>
  class KSurfaceAction<Action,Length<KBasisTypes>::value,boundaryID,shapeID>
  {
  public:
    static void ActOnSurfaceType(const KSurfaceID& anID,Action&)
    {
      std::cout<<"Unable to find a basis whose ID matches the ID in question ("<<anID.BasisID<<")."<<std::endl;
      return;
    }
  };

/**
* @class KSurfaceGeneratorAction
*
* @brief A class for generating surfaces.
*
* KSurfaceGeneratorAction is a surface action for generating policy-specified
* surfaces.
*
* @author T.J. Corona
*/

  class KSurfaceGeneratorAction : public KSurfaceAction<KSurfaceGeneratorAction>
  {
  public:
    template <typename Surface>
    void PerformAction(Type2Type<Surface>)
    {
      fSurfacePrimitive = new Surface();
    }

    KSurfacePrimitive* GetSurfacePrimitive() { return fSurfacePrimitive; }

  private:
    KSurfacePrimitive* fSurfacePrimitive;
  };

  class KSurfaceGenerator
  {
  public:
    static KSurfacePrimitive* GenerateSurface(KSurfaceID& anID)
    {
      KSurfaceGeneratorAction surfaceGeneratorAction;
      KSurfaceAction<KSurfaceGeneratorAction>::ActOnSurfaceType(anID,surfaceGeneratorAction);
      return surfaceGeneratorAction.GetSurfacePrimitive();
    }
  };

/**
* @class KSurfaceSize
*
* @brief A class for determining the size of a specific policy subset of a
* surface.
*
* KSurfaceSize provides the ability to measure the size of an associated policy
* as it is applied to a surface.
*
* @author T.J. Corona
*/

  template <class SurfacePolicy>
  class KSurfaceSize
  {
  public:
    template <class Policy>
    void PerformAction(Type2Type<Policy>)
    {
      Policy* policy = static_cast<Policy*>(fSurface);
      fTypeCounter << *policy;
    }

    template <class Streamed>
    void PreStreamOutAction(const Streamed&) {}
    template <class Streamed>
    void PostStreamOutAction(const Streamed&) {}

    void SetSurface(SurfacePolicy* s) { fSurface = s; }
    unsigned int size() const { return fTypeCounter.NumberOfTypes(); }
    void Reset() { fTypeCounter.Reset(); }

  private:
    KFundamentalTypeCounter fTypeCounter;
    SurfacePolicy* fSurface;
  };

/**
* @class KSurfaceStreamerAction
*
* @brief A class for streaming in and out of a surface.
*
* KSurfaceStreamerAction is a surface action for dispatching the stream pattern
* to the appropriate Basis, Boundary and Shape policy combination.  It
* facilitates the ability to stream a surface via a reference to its base,
* KSurfacePrimitive.
*
* @author T.J. Corona
*/

  template <typename Stream,bool inStream>
  class KSurfaceStreamerAction : public KSurfaceAction<KSurfaceStreamerAction<Stream,inStream> >
  {
  public:
    template <typename Surface>
    void PerformAction(Type2Type<Surface>)
    {
      fStream = &(StreamSurface<Surface>(*fStream,*fSurfacePrimitive,Int2Type<inStream>()));
    }

    template <typename Surface>
    Stream& StreamSurface(Stream& s,const KSurfacePrimitive& sP,Int2Type<true>)
    {
      return s << static_cast<const Surface&>(sP);
    }

    template <typename Surface>
    Stream& StreamSurface(Stream& s,const KSurfacePrimitive& sP,Int2Type<false>)
    {
      KSurfacePrimitive& sP_ = const_cast<KSurfacePrimitive&>(sP);
      return s >> static_cast<Surface&>(sP_);
    }

    void SetStream(Stream& s) { fStream = &s; }
    void SetSurfacePrimitive(const KSurfacePrimitive& sP) { fSurfacePrimitive = &sP; }
    Stream& GetStream() { return *fStream; }

  private:
    Stream* fStream;
    const KSurfacePrimitive* fSurfacePrimitive;
  };

  template <typename Stream,bool inStream>
  class KSurfaceStreamer
  {
  public:
    static Stream& StreamSurface(Stream& s,const KSurfacePrimitive& sP,const KSurfaceID& sID)
    {
      KSurfaceStreamerAction<Stream,inStream> surfaceStreamerAction;
      surfaceStreamerAction.SetStream(s);
      surfaceStreamerAction.SetSurfacePrimitive(sP);
      KSurfaceAction<KSurfaceStreamerAction<Stream,inStream> >::ActOnSurfaceType(sID,surfaceStreamerAction);
      return surfaceStreamerAction.GetStream();
    }

    static Stream& StreamSurface(Stream& s,const KSurfacePrimitive& sP)
    {
      return StreamSurface(s,sP,sP.GetID());
    }
  };

}

#endif /* KSURFACEACTION_DEF */
