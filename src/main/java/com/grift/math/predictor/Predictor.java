package com.grift.math.predictor;

import java.util.stream.IntStream;
import com.google.common.annotations.VisibleForTesting;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.ProbabilityVector;
import com.grift.math.real.Real;
import org.jetbrains.annotations.NotNull;
import org.springframework.stereotype.Component;

import static com.grift.math.real.Real.ONE;
import static com.grift.math.real.Real.ZERO;

@Component
public class Predictor {

    @NotNull
    private final ProbabilityVector.Factory vectorFactory;

    public Predictor(@NotNull SymbolIndexMap symbolIndexMap) {
        vectorFactory = new ProbabilityVector.Factory(symbolIndexMap.getImmutableCopy());
    }

    private static void makePrediction(@NotNull Real[] oldVec, @NotNull Real[] newVec, int min, int max, @NotNull Real[] prediction) {
        int mid = (max + min) / 2;
        int length = max - min + 1;

        if (length == 1) {
            prediction[mid] = ONE;
            return;
        }

        makePrediction(oldVec, newVec, min, mid, prediction);
        makePrediction(oldVec, newVec, mid + 1, max, prediction);

        Real[] weights = getElementWeights(oldVec, newVec, min, mid, max);

        IntStream.rangeClosed(min, max).forEach(i -> {
            Real weight = weights[i <= mid ? 0 : 1];
            prediction[i] = weight.times(prediction[i]);
        });
    }

    private static Real[] getElementWeights(@NotNull Real[] oldVec, @NotNull Real[] newVec, int min, int mid, int max) {
        Real[] oldWeights = sumHalves(oldVec, min, mid, max);
        Real[] newWeights = sumHalves(newVec, min, mid, max);
        return projectR2(oldWeights[0], newWeights[0]);
    }

    @NotNull
    private static Real[] sumHalves(Real[] arr, int min, int mid, int max) {
        Real[] weights = new Real[]{ZERO, ZERO};
        IntStream.rangeClosed(min, max).forEach(i -> {
            int index = i <= mid ? 0 : 1;
            weights[index] = weights[index].plus(arr[i]);
        });
        return normalize(weights);
    }

    /* project:  Solves for the steady-state vector (eigenvector with eigenvalue: 1) for a 2x2 right-stochastic system.
     *
     *       Consider a 2x2 matrix M such that its columns are probability vectors and M * o = n.
     *       It can be shown that L=1 must be one of its eigenvalues, and therefore there exists an
     *       eigenvector associated with it.  This function solves for
     *          a 2x2 probability vector parallel with that eigenvector.
     *
     * parameters:
     *    - o, n  are the old and new (respectively) probability vectors of dimension 2.
     *       probability vector => each component is 0 <= x <= 1  and the sum of the components of each vector is 1.
     *       i.e.: [0.6, 0.4] is a probability vector, [0.2, 0.3] is not and [-0.2, 1.2] is not.
     *
     *    - projection (output) is the by-reference return value which will contain the steady-state probability vector.
     *
     * returns:
     *    - false if either o or n are not probability vectors of dimension: 2
     *    - true and initializes projection on successful recovery of the vector.
     */
    @NotNull
    @VisibleForTesting()
    static Real[] projectR2(Real o, Real n) {
        if (o.equals(n)) {
            //Since o and n are both the first component of probability vectors of degree 2, we now know that
            //they must represent the same vector.  The most sensible prediction is that the system is static.
            //copy o into projection and be done with it because nothing is changing right now.
            return new Real[]{n, ONE.subtract(n)};
        }

        Real[] projection = calculate2DProjectionVector(o, n);

        /* I can prove inductively that if the vector o is not the zero vector (we checked with isProbabilityVector above eh?)
         * then the steady-state vector cannot be the zero vector.  Furthermore, its components must be non-negative.
         * So, we have a vector with non-negative components, not both zero.  Therefore the sum of the components must be positive.
         * We will scale the SSV by the reciprocal of the sum of its components to obtain a probability vector to which it is parallel.
         */
        return normalize(projection);
    }

    private static Real[] calculate2DProjectionVector(@NotNull Real v, @NotNull Real v1) {
        /* Sorry for the voodoo here.  I solved the 2x2 case myself and derived a general formula for
         * the steady state vector, up to one degree of freedom.  Let's call the free scalar s.
         * I then found formulae for the least-upper-bound on s (called max below)
         * and for the greatest-lower-bound on s (called min below) such that if s falls inside the inclusive interval [min, max],
         * we have a valid probability vector.  If s is chosen outside of that interval, one or more components will either become
         * negative or exceed 1 and is therefore illegal.  To integrate the components over all of these legal values of s is to
         * "add up" all feasible solutions to this problem.  So I did that too, manually, and finally derived a formula for that sum.
         * The following mystic incantation makes the vector <p1, p2> the sum of all feasible solutions to the problem (in one easy step!)
         */

        //<black_magic>
        Real[] projection = new Real[2];
        Real w = v.subtract(v1);
        Real x = ONE.subtract(v);
        Real y = ONE.plus(w);
        Real min = (x.subtract(v1)).divide(x);
        if (min.isNegative()) {
            min = ZERO;
        }
        Real max = (ONE.subtract(v1)).divide(x);
        if (max.isGreaterThan(ONE)) {
            max = ONE;
        }
        Real logOperand = (y.subtract(max)).divide(y.subtract(min));
        Real vwlog = v.times(w).times(logOperand.ln());
        Real range = max.subtract(min);
        Real p1 = v.times(range).plus(vwlog);
        Real p2 = x.times(range).subtract(vwlog);
        //</black_magic>

        projection[0] = p1;
        projection[1] = p2;
        return projection;
    }

    @NotNull
    private static Real[] normalize(Real... projection) {
        Real sum = ZERO;
        for (Real d : projection) {
            sum = sum.plus(d);
        }
        if (!sum.isZero()) {
            for (int i = 0; i < projection.length; i++) {
                projection[i] = projection[i].divide(sum);
            }
        }
        return projection;
    }

    @NotNull
    public ProbabilityVector getPrediction(@NotNull ProbabilityVector oldVec, @NotNull ProbabilityVector newVec) {
        if (oldVec.getDimension() != newVec.getDimension()) {
            throw new IllegalArgumentException("vectors of differing dimension");
        }
        Real[] predictionVector = new Real[oldVec.getDimension()];
        makePrediction(oldVec.getValues(), newVec.getValues(), 0, oldVec.getDimension() - 1, predictionVector);
        return vectorFactory.create(predictionVector);
    }
}
