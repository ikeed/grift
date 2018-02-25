package com.grift.math.predictor;

import java.util.Arrays;
import com.google.common.annotations.VisibleForTesting;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.ProbabilityVector;
import lombok.NonNull;
import org.jetbrains.annotations.NotNull;
import org.springframework.stereotype.Component;

@Component
public class Predictor {
    private static final double EPSILON = 0.000000001d;
    private final ProbabilityVector.Factory vectorFactory;

    public Predictor(@NonNull @NotNull SymbolIndexMap symbolIndexMap) {
        vectorFactory = new ProbabilityVector.Factory(symbolIndexMap.getImmutableCopy());
    }

    @NotNull
    @NonNull
    public ProbabilityVector getPrediction(@NonNull ProbabilityVector oldVec, @NonNull ProbabilityVector newVec) {
        if (oldVec.getDimension() != newVec.getDimension()) {
            throw new IllegalArgumentException("vectors of differing dimension");
        }
        double[] predictionVector = new double[oldVec.getDimension()];
        makePrediction(oldVec.getValues(), newVec.getValues(), 0, oldVec.getDimension() - 1, predictionVector);
        return vectorFactory.create(predictionVector);
    }

    private static void makePrediction(@NotNull @NonNull double[] oldVec, @NotNull @NonNull double[] newVec, int min, int max, @NotNull @NonNull double[] prediction) {
        int mid = (max + min) / 2;
        int length = max - min + 1;

        if (length == 1) {
            prediction[mid] = 1;
            return;
        }

        makePrediction(oldVec, newVec, min, mid, prediction);
        makePrediction(oldVec, newVec, mid + 1, max, prediction);

        double[] weights = getElementWeights(oldVec, newVec, min, max, mid);

        for (int i = min; i <= max; i++) {
            prediction[i] *= weights[i <= mid ? 0 : 1];
        }
        normalizeSection(prediction, min, max);
    }

    private static double[] getElementWeights(@NotNull @NonNull double[] oldVec, @NotNull @NonNull double[] newVec, int min, int max, int mid) {
        double[] oldWeights = sumHalves(oldVec, min, mid, max);
        double[] newWeights = sumHalves(newVec, min, mid, max);
        return projectR2(oldWeights, newWeights);
    }

    private static void normalizeSection(double[] arr, int min, int max) {
        double sum = 0;
        for (int i = min; i <= max; i++) {
            sum += arr[i];
        }
        for (int i = min; i <= max; i++) {
            arr[i] /= sum;
        }
    }

    @NotNull
    private static double[] sumHalves(double[] arr, int min, int mid, int max) {
        double[] weights = new double[]{0, 0};
        for (int i = min; i <= max; i++) {
            weights[i <= mid ? 0 : 1] += arr[i];
        }
        return weights;
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
    static double[] projectR2(double[] o, double[] n) {
        double[] projection = new double[2];

        if (isZero(o[0] - n[0])) {
            //Since o and n are both probability vectors of degree 2, we now know that
            //o and n must be the same vector.  The most sensible prediction is that the system is static.
            //copy o into projection and be done with it because nothing is changing right now.
            return normalize(Arrays.copyOf(o, 2));
        }

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
        double min = Math.max(0, ((1 - o[0] - n[0]) / (1 - o[0])));
        double max = Math.min(1, ((1 - n[0]) / (1 - o[0])));
        double log = Math.log(((1 + o[0] - n[0] - max) / (1 + o[0] - n[0] - min)));
        double p1 = o[0] * (max - min + (o[0] - n[0]) * log);
        double p2 = (max - min) - o[0] * (max - min) + (o[0] * n[0] - o[0] * o[0]) * log;
        //</black_magic>

        projection[0] = p1;
        projection[1] = p2;

        /* I can prove inductively that if the vector o is not the zero vector (we checked with isProbabilityVector above eh?)
         * then the steady-state vector cannot be the zero vector.  Furthermore, its components must be non-negative.
         * So, we have a vector with non-negative components, not both zero.  Therefore the sum of the components must be positive.
         * We will scale the SSV by the reciprocal of the sum of its components to obtain a probability vector to which it is parallel.
         */
        return normalize(projection);
    }

    private static boolean isZero(double v) {
        return Math.abs(v) < EPSILON;
    }

    @NotNull
    private static double[] normalize(double[] projection) {
        double sum = 0;
        for (double d : projection) {
            if (d < 0) {
                throw new IllegalArgumentException("negative element");
            }
            sum += d;
        }
        if (sum != 0) {
            for (int i = 0; i < projection.length; i++) {
                projection[i] /= sum;
            }
        }
        return projection;
    }
}
