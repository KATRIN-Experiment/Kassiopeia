#ifndef KEMTRANSFORMATION_DEF
#define KEMTRANSFORMATION_DEF

#include "KThreeVector_KEMField.hh"
#include "KTransitiveStreamer.hh"


namespace KEMField
{

/**
 * @class KEMTransformation
 *
 * @brief A class for rotating and reflecting three-vectors
 *
 * KEMTransformation is a class for performing rotations and reflections on
 * elements comprised of instances of KPosition and KDirection.  Transformations
 * are affected via the templated streamer pattern.
 *
 * @author T.J. Corona
 */

  class KEMTransformation : public KTransitiveStreamer<KEMTransformation>
  {
  public:
    KEMTransformation() {}
    ~KEMTransformation() {}

    template <class Object>
    void Transform(Object& object);

    virtual void Transform(KPosition&) = 0;
    virtual void Transform(KDirection&) = 0;

    template <class Streamed>
    void PreStreamInAction(Streamed&) {}
    template <class Streamed>
    void PostStreamInAction(Streamed&) {}
    template <class Streamed>
    void PreStreamOutAction(const Streamed&) {}
    template <class Streamed>
    void PostStreamOutAction(const Streamed&) {}

    void PostStreamInAction(KPosition& position) { Transform(position); }
    void PostStreamInAction(KDirection& direction) { Transform(direction); }
  };

  template <class Object>
  void KEMTransformation::Transform(Object& object)
  {
    object >> *this >> object;
  }

  class KRotation : public KEMTransformation
  {
  public:
    KRotation() : KEMTransformation(),
		  fOrigin(0.,0.,0.),
		  fAxis(0.,0.,1.),
		  fAngle(0.) {}
    ~KRotation() {}

    using KEMTransformation::Transform;

    void SetOrigin(const KPosition& origin) { fOrigin = origin; }
    void SetAxis(const KDirection& axis) { fAxis = axis; }
    void SetAngle(const double angle) { fAngle = angle; }

    void Transform(KPosition& position) { position.RotateAboutAxis(fOrigin,fAxis,fAngle); }
    void Transform(KDirection& direction) { direction.RotateAboutAxis(fOrigin,fAxis,fAngle); }

  private:
    KPosition fOrigin;
    KDirection fAxis;
    double fAngle;
  };

  class KTranslation : public KEMTransformation
  {
  public:
    KTranslation() : KEMTransformation(),
		  fTranslation(0.,0.,0.) {}
    ~KTranslation() {}

    using KEMTransformation::Transform;

    void SetTranslation(const KDirection& translation) { fTranslation = translation; }

    void Transform(KPosition& position) { position += fTranslation; }
    void Transform(KDirection&) {}

  private:
    KDirection fTranslation;
  };

  class KReflection : public KEMTransformation
  {
  public:
    KReflection() : KEMTransformation(),
		    fOrigin(0.,0.,0.),
		    fNormal(0.,0.,1.) {}
    ~KReflection() {}

    using KEMTransformation::Transform;

    void SetOrigin(const KPosition& origin) { fOrigin = origin; }
    void SetNormal(const KDirection& normal) { fNormal = normal; }

    void Transform(KPosition& position) { position.ReflectThroughPlane(fOrigin,fNormal); }
    void Transform(KDirection& direction) { direction.ReflectThroughPlane(fOrigin,fNormal); }

  private:
    KPosition fOrigin;
    KDirection fNormal;
  };

}

#endif /* KEMTRANSFORMATION_DEF */
