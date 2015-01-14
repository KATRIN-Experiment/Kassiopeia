#include "KGShapeRandom.hh"

#include <limits>

#include "KRandom.h"

namespace KGeoBag
{
  KThreeVector KGShapeRandom::Random(KGSurface* surface)
  {
    fRandom[0] = fRandom[1] = fRandom[2] = std::numeric_limits<double>::quiet_NaN();
    surface->AcceptNode(this);
    fRandom = surface->GetOrigin() + fRandom;

    return fRandom;
  }

  KThreeVector KGShapeRandom::Random(KGSpace* space)
  {
    fRandom[0] = fRandom[1] = fRandom[2] = std::numeric_limits<double>::quiet_NaN();
    space->AcceptNode(this);
    fRandom = space->GetOrigin() + fRandom;

    return fRandom;
  }

  KThreeVector KGShapeRandom::Random(std::vector<KGSurface*>& surfaces)
  {
    fRandom[0] = fRandom[1] = fRandom[2] = std::numeric_limits<double>::quiet_NaN();

    if(0 == surfaces.size()) {
		return fRandom;
	}

    double totalArea = 0;
    for(std::vector<KGSurface*>::const_iterator s = surfaces.begin();
    		s != surfaces.end(); ++s) {
    	KGSurface* surface = *s;

    	if(!surface->HasExtension<KGMetrics>()) {
    		surface->MakeExtension<KGMetrics>();
    	}

    	totalArea += surface->AsExtension<KGMetrics>()->GetArea();
    }

    KGSurface* selectedSurface = 0;
    double decision = Uniform(0, totalArea);
    for(std::vector<KGSurface*>::const_iterator s = surfaces.begin();
			s != surfaces.end(); ++s) {
		selectedSurface = *s;

		totalArea -= selectedSurface->AsExtension<KGMetrics>()->GetArea();

		if(decision > totalArea) {
			break;
		}
	}

    return Random(selectedSurface);
  }

  KThreeVector KGShapeRandom::Random(std::vector<KGSpace*>& spaces)
  {
    fRandom[0] = fRandom[1] = fRandom[2] = std::numeric_limits<double>::quiet_NaN();

    if(0 == spaces.size()) {
    	return fRandom;
    }

    double totalVolume = 0;
    for(std::vector<KGSpace*>::const_iterator v = spaces.begin();
    		v != spaces.end(); ++v) {
    	KGSpace* space = *v;

    	if(!space->HasExtension<KGMetrics>()) {
    		space->MakeExtension<KGMetrics>();
    	}

    	totalVolume += space->AsExtension<KGMetrics>()->GetVolume();
    }

    KGSpace* selectedSpace = 0;
    double decision = Uniform(0, totalVolume);
    for(std::vector<KGSpace*>::const_iterator v = spaces.begin();
			v != spaces.end(); ++v) {
		selectedSpace = *v;

		totalVolume -= selectedSpace->AsExtension<KGMetrics>()->GetVolume();

		if(decision > totalVolume) {
			break;
		}
	}

    return Random(selectedSpace);
  }

  double KGShapeRandom::Uniform(double min,double max) const
  {
    return katrin::KRandom::GetInstance().Uniform(min,max);
  }
}
