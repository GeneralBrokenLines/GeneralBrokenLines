/** \file
 *  Regression test for data index of kinks in composed trajectories.
 *
 *  Create a composed trajectory (2 components, 5 points each, scatterer on
 *  2nd point of each component), fit it and check number of kink data blocks
 *  returned by getScatResults for all points.
 *
 *  Before the fix the entry of scatDataIndex for the first point of each
 *  component but the first one was not set, resulting in wrong (huge) numData.
 *
 *  \remark Returns non-zero exit code in case of failure.
 */

#include <iostream>
#include <cstdlib>
#include "GblTrajectory.h"
#include "GblUtilities.h"

using namespace gbl;
using namespace Eigen;

int main() {
	const unsigned int nComp = 3;
    const unsigned int nPts = 5;
	const unsigned int scatPos = 1; // scatterer on 2nd point
	srand(4711);
	std::vector<std::pair<std::vector<GblPoint>, MatrixXd> > pointsAndTrafos;
	std::vector<unsigned int> expected; // expected number of kink data per label (index = label - 1)
	for (unsigned int iComp = 0; iComp < nComp; ++iComp) {
		std::vector<GblPoint> points;
		for (unsigned int iPt = 0; iPt < nPts; ++iPt) {
			// propagation in constant field (q/p is constrained by kinks)
			Matrix5d jac = (iPt > 0) ? gblSimpleJacobian(1., 1., 0.3) : Matrix5d::Identity();
			GblPoint point(jac);
			Vector2d meas(0.01 * (rand() / double(RAND_MAX) - 0.5),
					      0.01 * (rand() / double(RAND_MAX) - 0.5));
			Vector2d prec(1.e4, 1.e4);
			point.addMeasurement(meas, prec);
			unsigned int nExp = 0;
			if (iPt == scatPos) {
				Vector2d scat(0., 0.);
				// 2nd component: one dimension without precision
				Vector2d scatPrec(1.e4, (iComp == 1) ? 0. : 1.e4);
				point.addScatterer(scat, scatPrec);
				nExp = (iComp == 1) ? 1 : 2;
			}
			expected.push_back(nExp);
			points.push_back(point);
		}
		pointsAndTrafos.push_back(
				std::make_pair(points, MatrixXd(Matrix5d::Identity())));
	}

	GblTrajectory traj(pointsAndTrafos);
	double chi2, lostWeight;
	int ndf;
	if (traj.fit(chi2, ndf, lostWeight)) {
		std::cout << " Fit failed" << std::endl;
		return 1;
	}

	unsigned int nErr = 0;
	for (unsigned int label = 1; label <= nComp * nPts; ++label) {
		unsigned int numData = 0;
		VectorXd res(4), meas(4), resErr(4), dw(4); // large enough for any sane numData
		traj.getScatResults(label, numData, res, meas, resErr, dw);
		std::cout << " label " << label << ": numData " << numData
				<< ", expected " << expected[label - 1] << std::endl;
		if (numData != expected[label - 1])
			++nErr;
	}
	std::cout << (nErr ? " FAILED" : " OK") << std::endl;
	return nErr ? 1 : 0;
}