
#include "KGMetrics.hh"

namespace KGeoBag {
	/////////////////////////////////////////////////////////////////////
	/////////// Visitors for Calculation of Volumes and Areas ///////////
	/////////////////////////////////////////////////////////////////////

	void KGMetricsVolumeData::VisitVolume(KGVolume* /*aVolume*/) {
		// ToDo: Generic volume calculation
		metricsmsg( eWarning ) << "You are using the generic function to calculate volumes. "
				<< "But at the moment there is no algorithm implemented. "
				<< "Do you want to do this?" << eom;
		fVolume = std::numeric_limits<double>::quiet_NaN();
	}

	void KGMetricsVolumeData::VisitCylinderSpace(KGCylinderSpace* aVolume) {
		fVolume = KConst::Pi() * aVolume->R() * aVolume->R() * fabs(aVolume->Z1() - aVolume->Z2());
	}

	void KGMetricsVolumeData::VisitConeSpace(KGConeSpace* aVolume) {
		fVolume = 1./3. * KConst::Pi() * aVolume->RB() * aVolume->RB() * fabs(aVolume->ZA() - aVolume->ZB());
	}

	void KGMetricsVolumeData::VisitCutConeSpace(KGCutConeSpace* aVolume) {
		double R = aVolume->R1();
		double r = aVolume->R2();

		fVolume = fabs(aVolume->Z1() - aVolume->Z2()) / 3. * KConst::Pi() * (R * R + R * r + r * r);
	}

	void KGMetricsVolumeData::VisitBoxSpace(const KGBoxSpace* aVolume) {
		fVolume = fabs((aVolume->ZA() - aVolume->ZB())
				* (aVolume->XA() - aVolume->XB()) * (aVolume->YA() - aVolume->YB()));
	}

	void KGMetricsAreaData::VisitArea(KGArea* /*aArea*/) {
		// ToDo: Generic area calculation
		metricsmsg( eWarning ) << "You are using the generic function to calculate areas. "
				<< "But at the moment there is no algorithm implemented. "
				<< "Do you want to do this?" << eom;
		fArea = std::numeric_limits<double>::quiet_NaN();
	}

	void KGMetricsAreaData::VisitCylinderSurface(KGCylinderSurface* aArea) {
		fArea = 2 * KConst::Pi() * aArea->R() * (aArea->R() + fabs(aArea->Z1() - aArea->Z2()));
	}

	void KGMetricsAreaData::VisitConeSurface(KGConeSurface* aArea) {
		double h = fabs(aArea->ZA() - aArea->ZB());
		fArea = KConst::Pi() * aArea->RB() * (aArea->RB() + sqrt(aArea->RB() * aArea->RB() + h * h));
	}

	void KGMetricsAreaData::VisitCutConeSurface(KGCutConeSurface* aArea) {
		double h = fabs(aArea->Z1() - aArea->Z2());
		double R = aArea->R1();
		double r = aArea->R2();
		double m = sqrt((R - r) * (R - r) + h * h);

		fArea = KConst::Pi() * (R * R + r * r + (R + r) * m);
	}

	/////////////////////////////////////////////
	/////////// Other methods ///////////////////
	/////////////////////////////////////////////

	KGSpace* KGMetricsVolumeData::GetSpace() const {
		return fSpace;
	}

	double KGMetricsVolumeData::GetVolume() const {
		CalculateVolume();

		return fVolume;
	}

	void KGMetricsVolumeData::CalculateVolume() const {
		assert(0 != fSpace);
		if(fVolume < 0) {
			fSpace->AcceptNode(const_cast<KGMetricsVolumeData*>(this));
		}
	}

	double KGMetricsAreaData::GetArea() const {
		CalculateArea();
		return fArea;
	}

	void KGMetricsAreaData::CalculateArea() const {
		assert(0 != fSurface);
		if(fArea < 0) {
			fSurface->AcceptNode(const_cast<KGMetricsAreaData*>(this));
		}
	}
}
